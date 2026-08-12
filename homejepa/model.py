"""v1 model input/backbone: location axis = receptacle (DESIGN §9).

EpTensors turns an episode into tensors; location slots are the home's
receptacles (MAX_LOC), each embedded by (receptacle type, receptacle index,
room type, room index). Rooms remain as derived structure for rollup metrics
and NEG events."""
import json
import math

import numpy as np
import torch
import torch.nn as nn

from .world import CLASS_NAMES, RECEPT_TYPES, ROOM_TYPES, TICKS_PER_DAY

ETYPES = {"POS": 0, "NEG": 1, "SELF_PICK": 2, "SELF_DROP": 3}
QUERY_T, PAD_T = 4, 5
NCLS = len(CLASS_NAMES)
NRT = len(ROOM_TYPES)
NRECT = len(RECEPT_TYPES)
CIDX_CAP, OWN_NONE, RTIDX_NONE = 6, 4, 3
RECTIDX_CAP = 4
MAX_LOC = 32
TICK_SEC = 5


class EpTensors:
    def __init__(self, ep, max_events=256, aug_gap=0.0, aug_seed=0, noid=False):
        """noid=True removes cross-track instance identity from everything the
        model sees (DESIGN 11.1): tokens lose instance/owner embeddings, the
        query anchors on its REFERENCE TRACK only, window reservation and
        class-history features become class-level. GT/teacher stay privileged."""
        self.max_events = max_events
        self.noid = noid
        home = ep["home"]
        self.rooms = home["rooms"]
        self.recepts = home["recepts"]
        self.n_loc = len(self.recepts)
        assert self.n_loc <= MAX_LOC
        self.loc_pos = {rc["id"]: i for i, rc in enumerate(self.recepts)}
        room_by_id = {r["id"]: r for r in self.rooms}
        self.loc_room = {rc["id"]: rc["room"] for rc in self.recepts}
        self.loc_type = np.array([RECEPT_TYPES.index(rc["type"]) for rc in self.recepts], dtype=np.int64)
        self.loc_tidx = np.array([min(rc["tidx"], RECTIDX_CAP - 1) for rc in self.recepts], dtype=np.int64)
        self.loc_roomt = np.array([ROOM_TYPES.index(room_by_id[rc["room"]]["type"])
                                   for rc in self.recepts], dtype=np.int64)
        self.loc_roomti = np.array([min(room_by_id[rc["room"]]["tidx"], RTIDX_NONE - 1)
                                    for rc in self.recepts], dtype=np.int64)

        obj = {o["id"]: o for o in home["objects"]}
        self.home_rec = {o["id"]: self.loc_pos.get(o.get("home_recept"), -1)
                         for o in home["objects"]}
        ev = ep["events"]
        n = len(ev)
        self.et = np.zeros(n, dtype=np.int64)
        self.cls = np.full(n, NCLS, dtype=np.int64)
        self.cidx = np.full(n, CIDX_CAP, dtype=np.int64)
        self.own = np.full(n, OWN_NONE, dtype=np.int64)
        self.rt = np.full(n, NRT, dtype=np.int64)
        self.rti = np.full(n, RTIDX_NONE, dtype=np.int64)
        self.ert = np.full(n, NRECT, dtype=np.int64)
        self.erti = np.full(n, RECTIDX_CAP, dtype=np.int64)
        self.t0 = np.zeros(n, dtype=np.int64)
        self.t1 = np.zeros(n, dtype=np.int64)
        self.cov = np.zeros(n, dtype=np.float32)
        self.ev_obj = np.full(n, -1, dtype=np.int64)
        self.ev_room = np.full(n, -1, dtype=np.int64)
        self.ev_rec = np.full(n, -1, dtype=np.int64)
        rt_by_id = {r["id"]: (ROOM_TYPES.index(r["type"]), min(r["tidx"], RTIDX_NONE - 1))
                    for r in self.rooms}
        for i, e in enumerate(ev):
            self.et[i] = ETYPES[e["type"]]
            self.t0[i], self.t1[i] = e["t0"], e["t1"]
            if "room" in e:
                self.rt[i], self.rti[i] = rt_by_id[e["room"]]
                self.ev_room[i] = e["room"]
            if "recept" in e:
                k = self.loc_pos[e["recept"]]
                self.ert[i] = self.loc_type[k]
                self.erti[i] = self.loc_tidx[k]
                self.ev_rec[i] = e["recept"]
            if "obj" in e:
                o = obj[e["obj"]]
                self.ev_obj[i] = e["obj"]
                self.cls[i] = CLASS_NAMES.index(o["cls"])
                self.cidx[i] = min(o["cidx"], CIDX_CAP - 1)
                self.own[i] = o["owner"] if o["owner"] is not None else OWN_NONE
            if e["type"] == "NEG":
                self.cov[i] = e.get("cov", 0.0)
        if noid:
            self.cidx[:] = CIDX_CAP
            self.own[:] = OWN_NONE
        know = ETYPES["POS"], ETYPES["SELF_DROP"]
        self.obj_ev = {}
        for i in range(n):
            if self.ev_obj[i] >= 0:
                self.obj_ev.setdefault(int(self.ev_obj[i]), []).append(i)
        self.obj_know = {}
        for oid, js in self.obj_ev.items():
            self.obj_know[oid] = [j for j in js if self.et[j] in know]
        self.cls_ev = {}
        for i in range(n):
            if self.ev_obj[i] >= 0:
                self.cls_ev.setdefault(int(self.cls[i]), []).append(i)
        self.cls_self = {}
        for i in range(n):
            if self.et[i] in (ETYPES["SELF_PICK"], ETYPES["SELF_DROP"]):
                self.cls_self.setdefault(int(self.cls[i]), []).append(i)
        self.room_scan = {}
        for i in range(n):
            r = int(self.ev_room[i])
            if r >= 0:
                self.room_scan.setdefault(r, []).append(int(self.t1[i]))
        self.room_scan = {k: np.array(v) for k, v in self.room_scan.items()}

        self.gt_tl = {g["obj"]: g["moves"] for g in ep.get("gt", [])}
        self.T = ep["days"] * TICKS_PER_DAY
        # glance sets (combination observations) -- gt_place is privileged
        # (purity diagnostics only) and must never reach model features
        self.glances = ep.get("glances", [])
        self.gl_t = np.array([g["t1"] for g in self.glances], dtype=np.int64)

        self.queries = []
        aug_rng = np.random.RandomState(aug_seed + len(ev))
        for q in ep["queries"]:
            self.queries.append(self._build_query(q, obj))
            if aug_gap > 0 and aug_rng.rand() < aug_gap:
                m = int(aug_rng.choice([1, 2, 4]))
                own_pos = [j for j in self.obj_ev.get(q["obj"], [])
                           if self.et[j] == ETYPES["POS"] and self.t1[j] < q["qt"]]
                if len(own_pos) > m:
                    aq = self._build_query(q, obj, excl=frozenset(own_pos[-m:]))
                    aq["meta"] = dict(q, aug=1)
                    self.queries.append(aq)

    def _build_query(self, q, obj, excl=None):
        o = obj[q["obj"]]
        n_same = sum(1 for x in obj.values() if x["cls"] == o["cls"])
        idxs = self._window(q["qt"], q["obj"], excl=excl)
        if self.noid:
            # anchor = the reference track itself, advanced only by later
            # SELF_DROPs of the same class (your own hands identify what they
            # placed). Later POS of the class is NOT trusted as the anchor --
            # it may be a different instance; association is the model's job.
            anchor_rec, anchor_t = q["last_recept"], q["last_t"]
            for j in self.cls_ev.get(CLASS_NAMES.index(obj[q["obj"]]["cls"]), []):
                if self.t1[j] >= q["qt"] or self.ev_rec[j] < 0:
                    continue
                if self.et[j] == ETYPES["SELF_DROP"] and self.t1[j] > anchor_t:
                    anchor_rec, anchor_t = int(self.ev_rec[j]), int(self.t1[j])
        else:
            anchor_rec, anchor_t = self._last_known(q["obj"], q["qt"], excl=excl)
            if anchor_rec is None:
                anchor_rec, anchor_t = q["last_recept"], q["last_t"]
        ref_idx = -1
        for j in self.obj_ev.get(q["obj"], []):
            if self.et[j] == ETYPES["POS"] and self.t1[j] == q["last_t"]:
                ref_idx = j
                break
        ctx_fut = idxs if excl is None else self._window(q["qt"], q["obj"])
        return dict(
            qt=q["qt"], idxs=idxs, gt=self.loc_pos[q["gt_recept"]],
            futlbl=self._future_labels(q["obj"], q["qt"]),
            anchor=self.loc_pos[anchor_rec], anchor_t=anchor_t,
            fut=self._future_window(q["qt"], q["obj"], ctx_fut, k=1,
                                    gt_slot=self.loc_pos[q["gt_recept"]]),
            fut_ok1=self._fut_ok,
            fut2=self._future_window(q["qt"], q["obj"], ctx_fut, k=2,
                                     gt_slot=self.loc_pos[q["gt_recept"]]),
            fut_ok2=self._fut_ok,
            fut4=self._future_window(q["qt"], q["obj"], ctx_fut, k=4,
                                     gt_slot=self.loc_pos[q["gt_recept"]]),
            fut_ok4=self._fut_ok,
            absent=(self._no_absence() if excl else
                    self._absence_at(self.loc_pos[anchor_rec], anchor_t,
                                     CLASS_NAMES.index(o["cls"]), q["qt"])),
            hist=self._loc_hist(q["obj"], q["qt"], excl=excl),
            loc_feat=self._loc_feat(q["obj"], q["qt"], excl=excl),
            gl_feat=(self._no_glance() if excl else
                     self._glance_feat(o["cls"], q["qt"])),
            ref_idx=ref_idx,
            qcls=CLASS_NAMES.index(o["cls"]),
            qcidx=CIDX_CAP if self.noid else min(o["cidx"], CIDX_CAP - 1),
            qown=OWN_NONE if self.noid else (o["owner"] if o["owner"] is not None else OWN_NONE),
            meta=dict(q, multi=int(n_same > 1)))

    FUT_H = {"1h": 720, "6h": 4320}

    def _future_labels(self, oid, qt):
        from bisect import bisect_right as _br
        ms = self.gt_tl.get(oid)
        out = {}
        if not ms:
            return {k: -1 for k in list(self.FUT_H) + ["mv6h", "nmove", "nsight"]}
        for name, h in self.FUT_H.items():
            t = qt + h
            if t > self.T:
                out[name] = -1
                continue
            i = _br(ms, [t - 1, 10 ** 9]) - 1
            out[name] = self.loc_pos.get(ms[max(0, i)][1], -1)
        t6 = qt + self.FUT_H["6h"]
        out["mv6h"] = -1 if t6 > self.T else int(any(qt <= m[0] < t6 for m in ms))
        nmv = next((m for m in ms if m[0] >= qt), None)
        out["nmove"] = self.loc_pos.get(nmv[1], -1) if nmv else -1
        nsg = next((j for j in self.obj_know.get(oid, []) if self.t1[j] >= qt), None)
        out["nsight"] = (self.loc_pos.get(int(self.ev_rec[nsg]), -1)
                         if nsg is not None else -1)
        # tidy-assistant downstreams: where does this object belong, and is it
        # currently out of place (both product-facing, learnable from history)
        hr = self.home_rec.get(oid, -1)
        out["tidy"] = hr
        i = _br(ms, [qt - 1, 10 ** 9]) - 1
        cur = self.loc_pos.get(ms[max(0, i)][1], -1)
        out["misplaced"] = -1 if (hr < 0 or cur < 0) else int(cur != hr)
        return out

    def _future_window(self, qt, oid, ctx_idxs, k=1, gt_slot=None):
        nxts = [j for j in self.obj_know.get(oid, []) if self.t1[j] >= qt]
        if not nxts:
            if gt_slot is not None:
                self._fut_ok = False
            return ctx_idxs
        nxt = nxts[min(k, len(nxts)) - 1]
        if gt_slot is not None:
            rec = int(self.ev_rec[nxt])
            self._fut_ok = rec >= 0 and self.loc_pos.get(rec, -9) == gt_slot
        tmax = int(self.t1[nxt])
        extra = [j for j in range(len(self.t1)) if qt <= self.t1[j] <= tmax]
        keep = {j for j in (set(ctx_idxs.tolist()) | set(extra[-64:]) | {nxt})
                if self.t1[j] <= tmax}
        idxs = sorted(keep)
        return np.array(idxs[-max(self.max_events, 1):], dtype=np.int64)

    NEVER = 1.0

    GLF = 4     # combination-attribute columns appended to loc_feat

    def _no_glance(self):
        """Consistent no-information glance features for aug copies: the
        suffix-truncation hides sightings from the event window, but glances
        aggregate them back in (CONCEPT_REVIEW 2.2 leak) -- so aug copies get
        the 'never observed' state instead."""
        f = np.zeros((MAX_LOC, self.GLF), dtype=np.float32)
        f[:, 0] = self.NEVER
        return f

    def _no_absence(self):
        f = np.zeros(5, dtype=np.float32)
        f[0] = self.NEVER
        return f

    def _glance_feat(self, cls_name, qt):
        """Per-location combination attributes from glance sets (id-free):
        f_seen / f_wit / f_miss / f_aff  (DESIGN 11.6)."""
        f = np.zeros((MAX_LOC, self.GLF), dtype=np.float32)
        f[:, 0] = self.NEVER
        hi = int(np.searchsorted(self.gl_t, qt, side="left"))
        last_seen = {}
        wit = {}
        miss = {}
        aff_n = {}
        cls_glances = 0
        for g in self.glances[:hi]:
            objs = g["objs"]
            furn = set(g["furn"])
            has_c = cls_name in objs
            witnesses = len(objs) - (objs.count(cls_name) if has_c else 0)
            if has_c:
                cls_glances += 1
                for ft in furn:
                    aff_n[ft] = aff_n.get(ft, 0) + 1
            if witnesses >= 2:
                for ft in furn:
                    last_seen[ft] = g["t1"]
                    wit[ft] = witnesses
                    miss[ft] = 0 if has_c else miss.get(ft, 0) + 1
        for k in range(self.n_loc):
            ft = self.recepts[k]["type"]
            if ft in last_seen:
                f[k, 0] = math.log1p((qt - last_seen[ft]) * TICK_SEC / 60.0) / 10.0
                f[k, 1] = min(1.0, wit[ft] / 8.0)
                f[k, 2] = min(1.0, miss[ft] / 6.0)
            f[k, 3] = aff_n.get(ft, 0) / max(1, cls_glances)
        return f

    def _loc_feat(self, oid, qt, excl=None):
        """Per-receptacle negative-evidence attributes: since the receptacle's
        ROOM was scanned; since the object was seen on THIS receptacle."""
        f = np.zeros((MAX_LOC, 3), dtype=np.float32)
        last_seen = {}
        if self.noid:
            src = [j for j in self.cls_ev.get(int(self.cls[self.obj_ev[oid][0]]), [])
                   if self.et[j] in (ETYPES["POS"], ETYPES["SELF_DROP"])] \
                if self.obj_ev.get(oid) else []
        else:
            src = self.obj_know.get(oid, [])
        for j in src:
            if self.t1[j] >= qt or (excl and j in excl):
                continue
            if self.ev_rec[j] >= 0:
                last_seen[self.loc_pos[int(self.ev_rec[j])]] = int(self.t1[j])
        for k in range(self.n_loc):
            rid = self.loc_room[self.recepts[k]["id"]]
            ts = self.room_scan.get(rid)
            if ts is not None:
                i = int(np.searchsorted(ts, qt, side="left"))
                f[k, 0] = (math.log1p((qt - ts[i - 1]) * TICK_SEC / 60.0) / 10.0
                           if i > 0 else self.NEVER)
            else:
                f[k, 0] = self.NEVER
            if k in last_seen:
                f[k, 1] = math.log1p((qt - last_seen[k]) * TICK_SEC / 60.0) / 10.0
                f[k, 2] = 1.0
            else:
                f[k, 1] = self.NEVER
        return f

    def _absence_at(self, slot, since_t, cls_i, qt):
        """Absence evidence for THIS anchor slot (instance, not type):
        [seen-recency, witnesses, miss-count, ever-missed].
        The gate reads this directly -- measurement showed the model kept
        answering the anchor on 75% of confirmed-absence queries."""
        from .world import CLASS_NAMES as _CN
        cname = _CN[cls_i]
        stype = self.recepts[slot]["type"] if slot < self.n_loc else None
        last, wit, miss, ppl = None, 0, 0, 0
        for g in self.glances:
            if g["t1"] >= qt:
                break
            if g["t1"] <= since_t or stype not in g["furn"]:
                continue
            others = [c for c in g["objs"] if c != cname]
            if len(others) < 1:
                continue
            last, wit = int(g["t1"]), len(others)
            ppl = max(ppl, int(g.get("people", 0)))
            if cname not in g["objs"]:
                miss += 1
        f = np.zeros(5, dtype=np.float32)
        f[0] = (math.log1p((qt - last) * TICK_SEC / 60.0) / 10.0) if last is not None else self.NEVER
        f[1] = min(1.0, wit / 6.0)
        f[2] = min(1.0, miss / 4.0)
        f[3] = 1.0 if miss > 0 else 0.0
        f[4] = min(1.0, ppl / 2.0)     # L1: someone was around this surface
        return f

    def _loc_hist(self, oid, qt, excl=None):
        h = np.zeros(MAX_LOC, dtype=np.float32)
        src = self.cls_ev.get(int(self.cls[self.obj_ev[oid][0]]), []) \
            if (self.noid and self.obj_ev.get(oid)) else self.obj_ev.get(oid, [])
        for j in src:
            if excl and j in excl:
                continue
            if self.et[j] == ETYPES["POS"] and self.t1[j] < qt and self.ev_rec[j] >= 0:
                h[self.loc_pos[int(self.ev_rec[j])]] += (self.t1[j] - self.t0[j] + 1)
        s = h.sum()
        return h / s if s > 0 else h

    def _last_known(self, oid, qt, excl=None):
        best = None
        for j in self.obj_know.get(oid, []):
            if excl and j in excl:
                continue
            if self.t1[j] < qt:
                best = j
            else:
                break
        return (int(self.ev_rec[best]), int(self.t1[best])) if best is not None else (None, None)

    def _window(self, qt, oid, own_budget=64, excl=None):
        if self.max_events <= 0:
            return np.zeros(0, dtype=np.int64)
        i0 = int(np.searchsorted(self.t1, qt, side="left"))
        span = [j for j in range(i0, min(len(self.t1), i0 + 400)) if self.t0[j] < qt]
        avail = list(range(0, i0)) + span
        if excl:
            avail = [j for j in avail if j not in excl]
        src = self.cls_ev.get(int(self.cls[self.obj_ev[oid][0]]) if self.obj_ev.get(oid) else -1, []) \
            if self.noid else self.obj_ev.get(oid, [])
        own = [j for j in src
               if (j < i0 or j in span) and not (excl and j in excl)][-own_budget:]
        keep = set(own)
        for j in reversed(avail):
            if len(keep) >= self.max_events:
                break
            keep.add(j)
        return np.array(sorted(keep), dtype=np.int64)


def load_split(files, max_events=256, limit=None, aug_gap=0.0, noid=False):
    eps = []
    for fp in files[:limit] if limit else files:
        eps.append(EpTensors(json.load(open(fp)), max_events, aug_gap=aug_gap, noid=noid))
    return eps


def make_batch(eps, samples, device, window="idxs"):
    B = len(samples)
    L = max(len(eps[e].queries[q][window]) for e, q in samples) + 1
    et = np.full((B, L), PAD_T, dtype=np.int64)
    cls = np.full((B, L), NCLS, dtype=np.int64)
    cidx = np.full((B, L), CIDX_CAP, dtype=np.int64)
    own = np.full((B, L), OWN_NONE, dtype=np.int64)
    rt = np.full((B, L), NRT, dtype=np.int64)
    rti = np.full((B, L), RTIDX_NONE, dtype=np.int64)
    ert = np.full((B, L), NRECT, dtype=np.int64)
    erti = np.full((B, L), RECTIDX_CAP, dtype=np.int64)
    sca = np.zeros((B, L, 7), dtype=np.float32)
    pos = np.zeros((B, L), dtype=np.int64)
    pad = np.ones((B, L), dtype=bool)
    loc_t = np.full((B, MAX_LOC), NRECT, dtype=np.int64)
    loc_ti = np.full((B, MAX_LOC), RECTIDX_CAP, dtype=np.int64)
    loc_rt = np.full((B, MAX_LOC), NRT, dtype=np.int64)
    loc_rti = np.full((B, MAX_LOC), RTIDX_NONE, dtype=np.int64)
    loc_mask = np.zeros((B, MAX_LOC), dtype=bool)
    loc_anchor = np.zeros((B, MAX_LOC), dtype=np.int64)
    loc_feat = np.zeros((B, MAX_LOC, 8), dtype=np.float32)
    gt = np.zeros(B, dtype=np.int64)
    anchor = np.zeros(B, dtype=np.int64)
    qdt = np.zeros((B, 7), dtype=np.float32)
    hist = np.zeros((B, MAX_LOC), dtype=np.float32)

    for b, (ei, qi) in enumerate(samples):
        ep = eps[ei]
        q = ep.queries[qi]
        idxs = q[window]
        k = len(idxs)
        qt = q["qt"]
        if k:
            t1c = np.minimum(ep.t1[idxs], qt - 1) if window == "idxs" else ep.t1[idxs]
            t0c = np.minimum(ep.t0[idxs], t1c)
            et[b, :k] = ep.et[idxs]
            cls[b, :k] = ep.cls[idxs]
            cidx[b, :k] = ep.cidx[idxs]
            own[b, :k] = ep.own[idxs]
            rt[b, :k] = ep.rt[idxs]
            rti[b, :k] = ep.rti[idxs]
            ert[b, :k] = ep.ert[idxs]
            erti[b, :k] = ep.erti[idxs]
            tod = (t1c % TICKS_PER_DAY) / TICKS_PER_DAY * 2 * math.pi
            sca[b, :k, 0] = np.sin(tod)
            sca[b, :k, 1] = np.cos(tod)
            sca[b, :k, 2] = t1c / TICKS_PER_DAY / 4.0
            sca[b, :k, 3] = np.log1p((t1c - t0c) * TICK_SEC / 60.0) / 6.0
            dl = (qt - t1c) * TICK_SEC / 60.0
            sca[b, :k, 4] = np.sign(dl) * np.log1p(np.abs(dl)) / 8.0
            sca[b, :k, 5] = ep.cov[idxs]
            if getattr(ep, "noid", False):
                sca[b, :k, 6] = (idxs == q.get("ref_idx", -2)).astype(np.float32)
            else:
                sca[b, :k, 6] = (ep.ev_obj[idxs] == q["meta"]["obj"]).astype(np.float32)
        et[b, k] = QUERY_T
        cls[b, k] = q["qcls"]
        cidx[b, k] = q["qcidx"]
        own[b, k] = q["qown"]
        todq = (qt % TICKS_PER_DAY) / TICKS_PER_DAY * 2 * math.pi
        sca[b, k, 0] = math.sin(todq)
        sca[b, k, 1] = math.cos(todq)
        sca[b, k, 2] = qt / TICKS_PER_DAY / 4.0
        dt_anchor = (qt - q["anchor_t"]) * TICK_SEC
        sca[b, k, 4] = math.log1p(dt_anchor / 60.0) / 8.0
        pos[b, :k + 1] = np.arange(k, -1, -1)
        pad[b, :k + 1] = False
        nl = ep.n_loc
        loc_t[b, :nl] = ep.loc_type
        loc_ti[b, :nl] = ep.loc_tidx
        loc_rt[b, :nl] = ep.loc_roomt
        loc_rti[b, :nl] = ep.loc_roomti
        loc_mask[b, :nl] = True
        loc_anchor[b, q["anchor"]] = 1
        loc_feat[b, :, :3] = q["loc_feat"]
        loc_feat[b, :, 3] = q["hist"]
        loc_feat[b, :, 4:8] = q["gl_feat"]
        gt[b] = q["gt"]
        anchor[b] = q["anchor"]
        qdt[b, 0] = math.log1p(dt_anchor / 60.0) / 8.0
        qdt[b, 1] = min(1.0, dt_anchor / 86400.0)
        qdt[b, 2:7] = q["absent"]
        hist[b] = q["hist"]

    def T(x, dt=None):
        return torch.as_tensor(x, dtype=dt, device=device)
    return dict(et=T(et), cls=T(cls), cidx=T(cidx), own=T(own), rt=T(rt), rti=T(rti),
                ert=T(ert), erti=T(erti),
                sca=T(sca, torch.float32), pos=T(pos), pad=T(pad, torch.bool),
                loc_t=T(loc_t), loc_ti=T(loc_ti), loc_rt=T(loc_rt), loc_rti=T(loc_rti),
                loc_anchor=T(loc_anchor), loc_mask=T(loc_mask, torch.bool),
                loc_feat=T(loc_feat, torch.float32),
                gt=T(gt), anchor=T(anchor), qdt=T(qdt, torch.float32),
                hist=T(hist, torch.float32),
                room_mask=T(loc_mask, torch.bool))


class EventTransformer(nn.Module):
    """flat softmax over receptacle slots."""

    def __init__(self, d=128, layers=4, heads=4, max_pos=258, room_feats=True):
        super().__init__()
        self.room_feats = room_feats
        self.rf_proj = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d)) \
            if room_feats else None
        self.e_et = nn.Embedding(6, d)
        self.e_cls = nn.Embedding(NCLS + 1, d)
        self.e_cidx = nn.Embedding(CIDX_CAP + 1, d)
        self.e_own = nn.Embedding(OWN_NONE + 1, d)
        self.e_rt = nn.Embedding(NRT + 1, d)
        self.e_rti = nn.Embedding(RTIDX_NONE + 1, d)
        self.e_rec = nn.Embedding(NRECT + 1, d)
        self.e_reci = nn.Embedding(RECTIDX_CAP + 1, d)
        self.e_pos = nn.Embedding(max_pos, d)
        self.sca_proj = nn.Sequential(nn.Linear(7, d), nn.GELU(), nn.Linear(d, d))
        self.norm_in = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.1,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.loc_proj = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, d))
        self.q_head = nn.Linear(d, d)

    def encode(self, b):
        x = (self.e_et(b["et"]) + self.e_cls(b["cls"]) + self.e_cidx(b["cidx"]) +
             self.e_own(b["own"]) + self.e_rt(b["rt"]) + self.e_rti(b["rti"]) +
             self.e_rec(b["ert"]) + self.e_reci(b["erti"]) +
             self.e_pos(b["pos"]) + self.sca_proj(b["sca"]))
        h = self.enc(self.norm_in(x), src_key_padding_mask=b["pad"])
        qpos = (~b["pad"]).sum(1) - 1
        return h[torch.arange(h.size(0), device=h.device), qpos]

    def rooms(self, b):
        re = self.loc_proj(torch.cat([self.e_rec(b["loc_t"]), self.e_reci(b["loc_ti"]),
                                      self.e_rt(b["loc_rt"]), self.e_rti(b["loc_rti"])], -1))
        return re + self.rf_proj(b["loc_feat"]) if self.room_feats else re

    def forward(self, b):
        hq = self.q_head(self.encode(b))
        logits = torch.einsum("bd,brd->br", hq, self.rooms(b)) / math.sqrt(hq.size(-1))
        return logits.masked_fill(~b["loc_mask"], -1e9)

    def log_prob(self, b):
        return torch.log_softmax(self.forward(b), -1)


class TwoHeadEventTransformer(EventTransformer):
    """P(loc) = (1-g)*onehot(anchor) + g*softmax_{loc != anchor}."""

    def __init__(self, d=128, layers=4, heads=4, max_pos=258, room_feats=True):
        super().__init__(d, layers, heads, max_pos, room_feats)
        self.e_anchor = nn.Embedding(2, d)
        self.qdt_proj = nn.Sequential(nn.Linear(7, d), nn.GELU(), nn.Linear(d, d))
        # absence evidence goes straight into the gate, not only through the
        # shared trunk -- otherwise the 82% "stayed" prior drowns it out
        self.gate = nn.Sequential(nn.Linear(2 * d + 5, d), nn.GELU(), nn.Linear(d, 1))
        self.aux_head = nn.Linear(d, d)

    def rooms(self, b):
        re = super().rooms(b)
        return re + self.e_anchor(b["loc_anchor"])

    def _main_from(self, b, h, re):
        anchor_emb = re.gather(
            1, b["anchor"].view(-1, 1, 1).expand(-1, 1, re.size(-1))).squeeze(1)
        gate_logit = self.gate(torch.cat([h, anchor_emb, b["qdt"][:, 2:7]], -1)).squeeze(-1)
        hq = self.q_head(h)
        logits = torch.einsum("bd,brd->br", hq, re) / math.sqrt(hq.size(-1))
        anchor_1h = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            1, b["anchor"].view(-1, 1), True)
        logits = logits.masked_fill(~b["loc_mask"] | anchor_1h, -1e9)
        log_g = torch.nn.functional.logsigmoid(gate_logit).unsqueeze(-1)
        log_1mg = torch.nn.functional.logsigmoid(-gate_logit).unsqueeze(-1)
        lp_other = log_g + torch.log_softmax(logits, -1)
        return torch.where(anchor_1h, log_1mg.expand_as(lp_other), lp_other) \
                    .masked_fill(~b["loc_mask"], -1e9)

    def _aux_from(self, b, h, re):
        hq = self.aux_head(h)
        logits = torch.einsum("bd,brd->br", hq, re) / math.sqrt(hq.size(-1))
        return torch.log_softmax(logits.masked_fill(~b["loc_mask"], -1e9), -1)

    def aux_log_prob(self, b):
        h = self.encode(b) + self.qdt_proj(b["qdt"])
        return self._aux_from(b, h, self.rooms(b))

    def both(self, b):
        h = self.encode(b) + self.qdt_proj(b["qdt"])
        re = self.rooms(b)
        return self._main_from(b, h, re), self._aux_from(b, h, re)

    def all_heads(self, b):
        """(main, aux, gate_logit) from ONE encoder pass — training-loop use.
        both()+parts() re-encoded and doubled the step cost (P12 검토에서 적발)."""
        h = self.encode(b) + self.qdt_proj(b["qdt"])
        re = self.rooms(b)
        anchor_emb = re.gather(
            1, b["anchor"].view(-1, 1, 1).expand(-1, 1, re.size(-1))).squeeze(1)
        gate_logit = self.gate(torch.cat([h, anchor_emb, b["qdt"][:, 2:7]], -1)).squeeze(-1)
        return self._main_from(b, h, re), self._aux_from(b, h, re), gate_logit

    def parts(self, b):
        """(gate_logit, conditional log-softmax over non-anchor slots)."""
        h = self.encode(b) + self.qdt_proj(b["qdt"])
        re = self.rooms(b)
        anchor_emb = re.gather(
            1, b["anchor"].view(-1, 1, 1).expand(-1, 1, re.size(-1))).squeeze(1)
        gate_logit = self.gate(torch.cat([h, anchor_emb, b["qdt"][:, 2:7]], -1)).squeeze(-1)
        hq = self.q_head(h)
        logits = torch.einsum("bd,brd->br", hq, re) / math.sqrt(hq.size(-1))
        anchor_1h = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            1, b["anchor"].view(-1, 1), True)
        logits = logits.masked_fill(~b["loc_mask"] | anchor_1h, -1e9)
        return gate_logit, torch.log_softmax(logits, -1)

    def log_prob(self, b):
        h = self.encode(b) + self.qdt_proj(b["qdt"])
        return self._main_from(b, h, self.rooms(b))

    def forward(self, b):
        return self.log_prob(b)

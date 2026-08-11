"""Episode simulation v1: receptacle-level state (DESIGN §9).

Location = receptacle (surface). Rooms are derived (recept -> room). The old
zone abstraction is gone: receptacle positions ARE the FOV geometry. Within-
room churn (counter -> sink) now exists, matching HOMER+'s ~13:1 furniture:
room move ratio."""
import math
from bisect import bisect_right

from .world import OUT, SIZE_PDET
from .routines import (ACTIVITIES, CLASS_CHAIN, DISH_CLASSES, GO_OUT_TAKES,
                       WITHIN_SHIFT, build_schedules)

FOV_HALF = math.radians(45.0)
POS_GAP = 36                       # ticks (3 min) merge for flickering sightings
TICK_SEC = 5
TICKS_PER_MIN = 12

DTBINS = [(0, 600, "<10m"), (600, 3600, "10-60m"), (3600, 21600, "1-6h"), (21600, 10 ** 12, ">6h")]
QUERY_W = {"phone": 3.0, "keys": 2.5, "wallet": 2.5, "earbuds": 2.0, "laptop": 2.0,
           "bag": 1.5, "glasses": 1.2, "charger": 1.2, "tablet": 1.2, "remote": 1.0,
           "book": 1.0, "watch": 1.0, "nail_clipper": 0.8, "medicine": 0.8, "cup": 0.8,
           "notebook": 0.6, "water_bottle": 0.6, "pen": 0.5, "snack_box": 0.5,
           "scissors": 0.5, "umbrella": 0.5, "hair_dryer": 0.4, "speaker": 0.2, "plant": 0.1}


def dtbin(dt_sec):
    for lo, hi, name in DTBINS:
        if lo <= dt_sec < hi:
            return name
    return ">6h"


class _Sim:
    def __init__(self, home, days, rng, queries_per_ep=120):
        self.home, self.days, self.rng, self.nq = home, days, rng, queries_per_ep
        self.T = days * 17280
        self.rooms = home["rooms"]
        self.recepts = home["recepts"]
        self.objs = home["objects"]
        self.obj_cur = {o["id"]: o["init_recept"] for o in self.objs}
        self.moves = {o["id"]: [(0, o["init_recept"])] for o in self.objs}
        self.gt_moves = []
        self.in_use = {}
        self.carrier = {}
        self.self_events = []
        self.fetch_visits = []     # (tick, recept)
        self.events = []

    # ---------- helpers ----------
    def rec_room(self, rec):
        return self.recepts[rec]["room"] if rec != OUT else OUT

    def obj_room(self, oid):
        return self.rec_room(self.obj_cur[oid])

    def room_recept(self, rid, prefs=None):
        cands = self.rooms[rid]["recepts"]
        if prefs:
            for pt in prefs:
                hit = [g for g in cands if self.recepts[g]["type"] == pt]
                if hit:
                    return hit[0]
        return self.rng.choice(cands)

    def floor_of(self, rid):
        return self.room_recept(rid, ["floor"])

    def _move(self, oid, t, rec, mover, reason):
        cur = self.obj_cur[oid]
        t = max(int(t), self.moves[oid][-1][0] + 1)
        if rec == cur:
            return
        self.moves[oid].append((t, rec))
        self.gt_moves.append(dict(t=t, obj=oid, src=cur, dst=rec,
                                  src_room=self.rec_room(cur), dst_room=self.rec_room(rec),
                                  mover=mover, reason=reason))
        self.obj_cur[oid] = rec
        if mover == 0:
            if cur != OUT:
                self.self_events.append(dict(t=t, type="SELF_PICK", obj=oid,
                                             recept=cur, room=self.rec_room(cur)))
            if rec != OUT:
                self.self_events.append(dict(t=t, type="SELF_DROP", obj=oid,
                                             recept=rec, room=self.rec_room(rec)))

    def _busy(self, oid, t):
        return (t < self.in_use.get(oid, -1)) or (oid in self.carrier)

    def _pick_instance(self, cls, aidx, t):
        cands = []
        for o in self.objs:
            if o["cls"] != cls or self.obj_cur[o["id"]] == OUT or self._busy(o["id"], t):
                continue
            w = 1.0 if o["owner"] == aidx else (0.6 if o["owner"] is None else 0.45)
            cands.append((o["id"], w))
        if not cands:
            return None
        u = self.rng.random() * sum(w for _, w in cands)
        for oid, w in cands:
            u -= w
            if u <= 0:
                return oid
        return cands[-1][0]

    # ---------- phase 1: interactions ----------
    def run_interactions(self, schedules):
        rng = self.rng
        acts = self.home.get("pack_acts", ACTIVITIES)
        self._acts = acts
        for segs in schedules:
            for s in segs:
                if s["room"] != OUT:
                    s["recept"] = self.room_recept(s["room"],
                                                   acts.get(s["act"], {}).get("recepts"))
        bounds = []
        for segs in schedules:
            for s in segs:
                bounds.append((s["t0"], 1, s))
                bounds.append((s["t1"], 0, s))
        bounds.sort(key=lambda x: (x[0], x[1]))
        seg_used = {}
        for t, kind, s in bounds:
            a = s["agent"]
            if kind == 0:
                if s["act"] == "out":
                    self._return_from_out(s)
                for oid in seg_used.pop(id(s), []):
                    if oid in self.carrier:
                        continue
                    o = self.objs[oid]
                    p_carry = 0.15 if a == 0 else 0.22
                    if rng.random() < p_carry:
                        # time-consistent state: a concurrent agent may have
                        # scheduled a FUTURE-dated move (tidy) that already
                        # updated obj_cur; if so, skip the attach instead of
                        # emitting a pick at a place the object has not
                        # reached yet (audit C.orphan_selfpick race)
                        rec_now = self.gt_recept(oid, s["t1"])
                        if rec_now != self.obj_cur[oid]:
                            continue
                        self.carrier[oid] = a
                        if a == 0:
                            self.self_events.append(dict(
                                t=s["t1"], type="SELF_PICK", obj=oid,
                                recept=rec_now, room=self.rec_room(rec_now)))
                        continue
                    act_after = self._acts.get(s["act"], {}).get("after", {}).get(o["cls"])
                    base_chain = self.home.get("pack_chain", CLASS_CHAIN).get(o["cls"])
                    chain = act_after if (act_after and (base_chain is None or rng.random() < 0.5)) \
                        else base_chain
                    if chain and isinstance(chain[0], list):
                        chain = [tuple(x) for x in chain]
                    if chain:
                        # furniture-lifecycle destination (dish/food cycles)
                        u = rng.random()
                        dest = "stay"
                        for name, pr in chain:
                            u -= pr
                            if u <= 0:
                                dest = name
                                break
                        if dest == "stay":
                            continue
                        if dest == "home":
                            dst = o["home_recept"]
                        else:
                            rid = self.rec_room(self.obj_cur[oid])
                            dst = self.room_recept(rid, [dest])
                        self._move(oid, s["t1"] + rng.randint(0, 12), dst, a,
                                   "chain:" + dest)
                        if a == 0:
                            self.fetch_visits.append((self.moves[oid][-1][0] + 1, dst))
                        continue
                    p_home = 0.2 + 0.5 * self.home["agents"][a]["tidiness"]
                    p_shift = 0.30 if o["cls"] in WITHIN_SHIFT else 0.12
                    r = rng.random()
                    if r < p_home:
                        self._move(oid, s["t1"] + rng.randint(0, 12), o["home_recept"],
                                   a, "return_home")
                        if a == 0:
                            self.fetch_visits.append((self.moves[oid][-1][0] + 1, o["home_recept"]))
                    elif r < p_home + p_shift:
                        rid = self.rec_room(self.obj_cur[oid])
                        dst = self.room_recept(rid, WITHIN_SHIFT.get(o["cls"]))
                        self._move(oid, s["t1"] + rng.randint(0, 12), dst, a, "shift_within")
                continue
            if s["act"] == "sleep":
                self._drop_carried(s, p_override=0.9)
                continue
            if s["act"] == "out":
                self._go_out(s)
                continue
            self._carried_follow(s)
            used = []
            for cls, p in acts[s["act"]].get("uses", {}).items():
                if rng.random() >= min(0.97, p * (1.35 if a != 0 else 1.0)):
                    continue
                oid = self._pick_instance(cls, a, t)
                if oid is None:
                    continue
                src = self.obj_cur[oid]
                if src != s["recept"]:
                    t_f = s["t0"] + rng.randint(0, 36)
                    if a == 0:
                        self.fetch_visits.append((t_f, src))
                    self._move(oid, t_f, s["recept"], a, "fetch:" + s["act"])
                self.in_use[oid] = s["t1"]
                used.append(oid)
            seg_used[id(s)] = used
            if s["act"] == "wash_dishes":
                self._wash_dishes(s)
            if s["act"] == "chores" and rng.random() < self.home["agents"][a]["tidy_daily"]:
                self._tidy(s)

    def _carried_follow(self, s):
        a, rng = s["agent"], self.rng
        for oid in [o for o, c in self.carrier.items() if c == a]:
            p_drop = 0.35 if self.objs[oid]["cls"] == "phone" else 0.6
            if rng.random() < p_drop:
                del self.carrier[oid]
                self._move(oid, s["t0"] + rng.randint(0, 12), s["recept"], a, "drop_carried")
            else:
                cur_room = self.obj_room(oid)
                if cur_room != s["room"]:
                    self._move(oid, s["t0"], self.floor_of(s["room"]), a, "carried")

    def _drop_carried(self, s, p_override):
        a, rng = s["agent"], self.rng
        for oid in [o for o, c in self.carrier.items() if c == a]:
            if rng.random() < p_override:
                del self.carrier[oid]
                self._move(oid, s["t0"] + rng.randint(0, 12), s["recept"], a, "drop_sleep")
            elif self.obj_room(oid) != s["room"]:
                self._move(oid, s["t0"], self.floor_of(s["room"]), a, "carried")

    def _go_out(self, s):
        a, rng = s["agent"], self.rng
        taken = []
        for cls, p in GO_OUT_TAKES.items():
            if rng.random() >= p:
                continue
            oid = None
            for o in self.objs:
                if o["cls"] == cls and o["owner"] == a and self.obj_cur[o["id"]] != OUT \
                        and not self._busy(o["id"], s["t0"]):
                    oid = o["id"]
                    break
            if oid is None:
                oid = self._pick_instance(cls, a, s["t0"])
            if oid is None:
                continue
            if a == 0:
                self.fetch_visits.append((s["t0"] - rng.randint(1, 24), self.obj_cur[oid]))
            self._move(oid, s["t0"], OUT, a, "go_out")
            taken.append(oid)
        for oid in [o for o, c in self.carrier.items() if c == a]:
            self._move(oid, s["t0"], OUT, a, "go_out_carried")
            taken.append(oid)
            del self.carrier[oid]
        s["taken"] = taken

    def _return_from_out(self, s):
        a, rng = s["agent"], self.rng
        ent = self.home["rooms_by_type"].get("entrance")
        liv = self.home["rooms_by_type"]["living_room"][0]
        bed_room = self.home["agents"][a]["bedroom"]
        for oid in s.get("taken", []):
            r = rng.random()
            if r < 0.45 and ent:
                dst = self.room_recept(ent[0], ["console", "shoe_cabinet"])
                self._move(oid, s["t1"] + rng.randint(0, 12), dst, a, "return_drop")
            elif r < 0.65:
                self._move(oid, s["t1"] + rng.randint(0, 12),
                           self.room_recept(liv), a, "return_drop")
            elif r < 0.80:
                self._move(oid, s["t1"] + rng.randint(0, 12),
                           self.room_recept(bed_room), a, "return_drop")
            else:
                self._move(oid, s["t1"], self.room_recept(liv), a, "return_drop")
                self.carrier[oid] = a
        s.pop("taken", None)

    def _wash_dishes(self, s):
        """Dish cycle completion: dishes at the sink (or left on kitchen
        surfaces) go back to the cupboard / their home."""
        a, rng = s["agent"], self.rng
        kitchen = self.rec_room(s["recept"])
        for o in self.objs:
            oid = o["id"]
            if o["cls"] not in DISH_CLASSES or self._busy(oid, s["t0"]):
                continue
            cur = self.obj_cur[oid]
            if cur == OUT or self.rec_room(cur) != kitchen:
                continue
            if self.recepts[cur]["type"] not in ("sink", "counter", "dining_table"):
                continue
            if rng.random() < 0.75:
                dst = self.room_recept(kitchen, ["cabinet_top"])
                if self.recepts[dst]["type"] != "cabinet_top":
                    dst = o["home_recept"]
                t_m = rng.randint(s["t0"], max(s["t0"] + 1, s["t1"] - 1))
                if a == 0:
                    self.fetch_visits.append((t_m, cur))
                self._move(oid, t_m, dst, a, "wash_dishes")

    def _tidy(self, s):
        a, rng = s["agent"], self.rng
        mis = [o["id"] for o in self.objs
               if self.obj_cur[o["id"]] not in (OUT, o["home_recept"])
               and not self._busy(o["id"], s["t0"])]
        rng.shuffle(mis)
        for oid in mis[:rng.randint(2, 5)]:
            t_m = rng.randint(s["t0"], max(s["t0"] + 1, s["t1"] - 1))
            if a == 0:
                self.fetch_visits.append((t_m, self.obj_cur[oid]))
            self._move(oid, t_m, self.objs[oid]["home_recept"], a, "tidy")
            if a == 0:
                self.fetch_visits.append((self.moves[oid][-1][0] + 1,
                                          self.objs[oid]["home_recept"]))

    # ---------- phase 2: observation ----------
    def run_observation(self, schedules):
        rng = self.rng
        T = self.T
        self.user_returns = [s["t1"] for s in schedules[0] if s["act"] == "out"]
        # L1 person presence: how many OTHER residents are in a given room at a
        # given tick. Identity and carried objects are NOT modeled -- person
        # detection at 1fps is reliable, person re-id is not (same stance as
        # object re-id removal, DESIGN 11.1).
        others = [[0] * T for _ in self.rooms]
        for segs in schedules[1:]:
            for sg in segs:
                if sg["room"] == OUT or sg["act"] == "sleep":
                    continue
                row = others[sg["room"]]
                for t in range(max(0, sg["t0"]), min(T, sg["t1"])):
                    row[t] += 1
        u_rec = [-1] * T
        awake = [False] * T
        for s in schedules[0]:
            rec = s.get("recept", -1) if s["room"] != OUT else -1
            aw = s["act"] not in ("sleep",)
            for t in range(max(0, s["t0"]), min(T, s["t1"])):
                u_rec[t] = rec
                awake[t] = aw
        for t in range(1, T):
            if u_rec[t] == -1 and not awake[t] and u_rec[t - 1] != -1 and not awake[t - 1]:
                u_rec[t] = u_rec[t - 1]
        for t, rec in self.fetch_visits:
            if 0 <= t < T and rec != OUT:
                u_rec[t] = rec
                awake[t] = True

        GL_W = 12                      # glance window: 1 min
        glances = []
        gl = None                      # [t0, room, furn Counter, obj Counter, place Counter]

        def close_gl():
            nonlocal gl
            if gl and (gl[2] or gl[3]):
                glances.append(dict(
                    t0=gl[0], t1=gl[5], room=gl[1],
                    furn=sorted(self.recepts[g]["type"] for g in gl[2]),
                    objs=sorted(self.objs[o]["cls"] for o in gl[3]),
                    people=gl[6],
                    gt_place=gl[4].most_common(1)[0][0] if gl[4] else -1))
            gl = None

        from collections import Counter as _C
        move_q = sorted((t, oid, rec) for oid, ms in self.moves.items() for t, rec in ms)
        cur_rec = {}
        room_objs = {r["id"]: set() for r in self.rooms}
        qi = 0
        theta = rng.uniform(-math.pi, math.pi)
        pos_run = {}      # oid -> [t0, last, recept]
        neg_run = None    # [room, t0, last, set(recepts)]
        pos_events, neg_events = [], []

        def close_pos(oid):
            r = pos_run.pop(oid, None)
            if r:
                pos_events.append(dict(t0=r[0], t1=r[1], type="POS", obj=oid,
                                       recept=r[2], room=self.rec_room(r[2])))

        def close_neg():
            nonlocal neg_run
            if neg_run:
                nrec = len(self.rooms[neg_run[0]]["recepts"])
                neg_events.append(dict(t0=neg_run[1], t1=neg_run[2], type="NEG",
                                       room=neg_run[0], nz=nrec,
                                       cov=round(len(neg_run[3]) / nrec, 3)))
                neg_run = None

        for t in range(T):
            while qi < len(move_q) and move_q[qi][0] < t:
                _, oid, rec = move_q[qi]
                prev = cur_rec.get(oid)
                if prev is not None and prev != OUT:
                    room_objs[self.rec_room(prev)].discard(oid)
                if rec != OUT:
                    room_objs[self.rec_room(rec)].add(oid)
                cur_rec[oid] = rec
                qi += 1
            if not awake[t] or u_rec[t] == -1:
                close_neg()
                close_gl()
                continue
            urec = u_rec[t]
            room = self.rec_room(urec)
            upos = self.recepts[urec]["pos"]
            theta += rng.gauss(0, 0.25)
            vis = set()
            for g in self.rooms[room]["recepts"]:
                rp = self.recepts[g]["pos"]
                dx, dy = rp[0] - upos[0], rp[1] - upos[1]
                d = math.hypot(dx, dy)
                if g == urec or d < 0.05:
                    vis.add(g)
                    continue
                da = (math.atan2(dy, dx) - theta + math.pi) % (2 * math.pi) - math.pi
                if abs(da) <= FOV_HALF:
                    vis.add(g)
            if neg_run and neg_run[0] == room and t - neg_run[2] <= 1:
                neg_run[2] = t
                neg_run[3] |= vis
            else:
                close_neg()
                neg_run = [room, t, t, set(vis)]
            if gl is None or gl[1] != room or t - gl[0] >= GL_W:
                close_gl()
                gl = [t, room, set(), set(), _C(), t, 0]
            gl[5] = t
            gl[4][urec] += 1
            gl[6] = max(gl[6], others[room][t])
            for g in vis:
                if rng.random() < 0.9:
                    gl[2].add(g)
            for oid in room_objs[room]:
                if oid in self.carrier:
                    continue
                g = cur_rec.get(oid)
                if g is None or g not in vis:
                    if oid in pos_run and t - pos_run[oid][1] > POS_GAP:
                        close_pos(oid)
                    continue
                rp = self.recepts[g]["pos"]
                d = math.hypot(rp[0] - upos[0], rp[1] - upos[1])
                p = SIZE_PDET[self.objs[oid]["size"]] * (1.0 if d < 0.35 else (0.85 if d < 0.7 else 0.65))
                if rng.random() < p:
                    gl[3].add(oid)
                    r = pos_run.get(oid)
                    if r and r[2] == g and t - r[1] <= POS_GAP:
                        r[1] = t
                    else:
                        close_pos(oid)
                        pos_run[oid] = [t, t, g]
                elif oid in pos_run and t - pos_run[oid][1] > POS_GAP:
                    close_pos(oid)
        for oid in list(pos_run):
            close_pos(oid)
        close_neg()
        close_gl()
        self.glances = glances

        self.events = pos_events + neg_events + \
            [dict(t0=e["t"], t1=e["t"], type=e["type"], obj=e["obj"],
                  recept=e["recept"], room=e["room"]) for e in self.self_events]
        self.events.sort(key=lambda e: (e["t1"], e["t0"], e["type"]))
        self.awake = awake
        self.pos_index = {}
        for e in pos_events:
            self.pos_index.setdefault(e["obj"], []).append((e["t0"], e["t1"], e["recept"]))
        for v in self.pos_index.values():
            v.sort()

    # ---------- phase 3: queries ----------
    def gt_recept(self, oid, t):
        """state(t) = receptacle after all moves STRICTLY before t."""
        ms = self.moves[oid]
        i = bisect_right(ms, (t - 1, 10 ** 9)) - 1
        return ms[max(0, i)][1]

    def make_query(self, oid, qt):
        runs = self.pos_index.get(oid)
        if not runs:
            return None
        i = bisect_right(runs, (qt, 10 ** 12, 10 ** 9)) - 1
        if i < 0:
            return None
        t0, t1, rec = runs[i]
        if t0 <= qt <= t1 + 1:
            return None
        g = self.gt_recept(oid, qt)
        if g == OUT:
            return None
        dt = (qt - t1) * TICK_SEC
        movers = [m["mover"] for m in self.gt_moves
                  if m["obj"] == oid and t1 <= m["t"] < qt]
        return dict(qt=qt, obj=oid, gt_recept=g, gt_room=self.rec_room(g),
                    last_recept=rec, last_room=self.rec_room(rec), last_t=t1,
                    dt=dt, moved=int(g != rec),
                    moved_room=int(self.rec_room(g) != self.rec_room(rec)),
                    mover=(movers[-1] if movers else None),
                    n_moves=len(movers), dtbin=dtbin(dt))

    def sample_queries(self):
        rng = self.rng
        start = 6 * 3600 // TICK_SEC
        rets = [t for t in self.user_returns if t > start]
        cands = []
        tries = 0
        while tries < 600:
            tries += 1
            if rets and rng.random() < 0.4:
                qt = min(self.T - 1, rng.choice(rets) + rng.randint(0, 120 * TICKS_PER_MIN))
            else:
                qt = rng.randint(start, self.T - 1)
            if not self.awake[qt]:
                continue
            for o in self.objs:
                oid = o["id"]
                runs = self.pos_index.get(oid)
                if not runs:
                    continue
                i = bisect_right(runs, (qt, 10 ** 12, 10 ** 9)) - 1
                if i < 0:
                    continue
                t0, t1, rec = runs[i]
                if t0 <= qt <= t1 + 1:
                    continue
                if self.gt_recept(oid, qt) == OUT:
                    continue
                cands.append((qt, oid, t1))
        rng.shuffle(cands)
        seen = set()
        queries = []
        for qt, oid, last_t in cands:
            dt_sec = (qt - last_t) * TICK_SEC
            if dt_sec < 300:
                continue
            w_dt = 0.25 + 0.75 * min(1.0, dt_sec / 7200.0)
            if rng.random() >= (QUERY_W.get(self.objs[oid]["cls"], 1.0) / 2.0) * w_dt:
                continue
            key = (oid, qt // (30 * TICKS_PER_MIN))
            if key in seen:
                continue
            seen.add(key)
            q = self.make_query(oid, qt)
            if q:
                queries.append(q)
            if len(queries) >= self.nq:
                break
        return queries


def simulate_episode(home, days, rng, queries_per_ep=120):
    sim = _Sim(home, days, rng, queries_per_ep)
    schedules = build_schedules(home, days, rng)
    sim.run_interactions(schedules)
    sim.run_observation(schedules)
    queries = sim.sample_queries()
    return dict(
        home=dict(id=home["id"], n_agents=home["n_agents"],
                  rooms=[dict(id=r["id"], type=r["type"], tidx=r["tidx"],
                              recepts=r["recepts"]) for r in home["rooms"]],
                  recepts=[dict(id=rc["id"], type=rc["type"], tidx=rc["tidx"],
                                room=rc["room"], pos=rc["pos"]) for rc in home["recepts"]],
                  objects=[dict(id=o["id"], cls=o["cls"], cidx=o["cidx"], owner=o["owner"],
                                size=o["size"], home_recept=o["home_recept"])
                           for o in home["objects"]]),
        days=days,
        events=sim.events,
        glances=sim.glances,
        gt=[dict(obj=oid, moves=ms) for oid, ms in sorted(sim.moves.items())],
        gt_moves=sim.gt_moves,
        queries=queries)

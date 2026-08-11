"""v1 non-learned baselines at receptacle level."""
import json
import math
import os
from bisect import bisect_right
from collections import defaultdict

from .world import OUT, CLASS_NAMES

SMOOTH = 0.02


def _norm(d):
    t = sum(d.values())
    n = max(1, len(d))
    if t <= 0:
        return {k: 1.0 / n for k in d}
    return {k: v / t for k, v in d.items()}


def _smoothed_onehot(locs, hot):
    return {r: (1.0 - SMOOTH) + SMOOTH / len(locs) if r == hot else SMOOTH / len(locs)
            for r in locs}


def type_dist_to_locs(home, typew):
    cnt = defaultdict(int)
    for rc in home["recepts"]:
        cnt[rc["type"]] += 1
    out = {rc["id"]: typew.get(rc["type"], 0.0) / cnt[rc["type"]] for rc in home["recepts"]}
    if sum(out.values()) <= 0:
        out = {rc["id"]: 1.0 for rc in home["recepts"]}
    return _norm(out)


class EpisodeCtx:
    def __init__(self, ep):
        self.ep = ep
        self.home = ep["home"]
        self.locs = [rc["id"] for rc in ep["home"]["recepts"]]
        self.ltype = {rc["id"]: rc["type"] for rc in ep["home"]["recepts"]}
        self.lroom = {rc["id"]: rc["room"] for rc in ep["home"]["recepts"]}
        self.obj = {o["id"]: o for o in ep["home"]["objects"]}
        self.pos_runs = defaultdict(list)
        self.know_runs = defaultdict(list)
        for e in ep["events"]:
            if e["type"] == "POS":
                self.pos_runs[e["obj"]].append((e["t0"], e["t1"], e["recept"]))
                self.know_runs[e["obj"]].append((e["t0"], e["t1"], e["recept"]))
            elif e["type"] == "SELF_DROP":
                self.know_runs[e["obj"]].append((e["t0"], e["t1"], e["recept"]))
        for v in self.pos_runs.values():
            v.sort()
        for v in self.know_runs.values():
            v.sort()

    def last_known(self, oid, qt):
        runs = self.know_runs.get(oid, [])
        i = bisect_right(runs, (qt, 10 ** 12, 10 ** 9)) - 1
        return (runs[i][2], runs[i][1]) if i >= 0 else (None, None)

    def pos_before(self, oid, qt):
        runs = self.pos_runs.get(oid, [])
        i = bisect_right(runs, (qt, 10 ** 12, 10 ** 9))
        return runs[:i]


def fit_stats(train_files):
    occ = defaultdict(lambda: defaultdict(float))
    trans = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    samples = defaultdict(list)
    for fp in train_files:
        ep = json.load(open(fp))
        ltype = {rc["id"]: rc["type"] for rc in ep["home"]["recepts"]}
        ocls = {o["id"]: o["cls"] for o in ep["home"]["objects"]}
        T = ep["days"] * 17280
        for g in ep["gt"]:
            ms = g["moves"]
            for i, (t, rec) in enumerate(ms):
                if rec == OUT:
                    continue
                t_end = ms[i + 1][0] if i + 1 < len(ms) else T
                occ[ocls[g["obj"]]][ltype[rec]] += (t_end - t)
        for m in ep["gt_moves"]:
            if m["src"] == OUT or m["dst"] == OUT:
                continue
            trans[ocls[m["obj"]]][ltype[m["src"]]][ltype[m["dst"]]] += 1.0
        for q in ep["queries"]:
            samples[ocls[q["obj"]]].append((q["dt"], q["moved"]))

    taus = {}
    grid = [600.0 * (1.35 ** i) for i in range(40)]
    for cls in CLASS_NAMES:
        ss = samples.get(cls, [])
        if not ss:
            taus[cls] = grid[-1]
            continue
        best, best_ll = grid[-1], -1e18
        for tau in grid:
            ll = 0.0
            for dt, moved in ss:
                ps = math.exp(-dt / tau)
                ll += math.log(max(1e-9, 1.0 - ps)) if moved else math.log(max(1e-9, ps))
            if ll > best_ll:
                best_ll, best = ll, tau
        taus[cls] = best

    prior = {c: _norm(dict(d)) for c, d in occ.items()}
    trans_out = {}
    glob = defaultdict(lambda: defaultdict(float))
    for c, rows in trans.items():
        trans_out[c] = {s: _norm(dict(d)) for s, d in rows.items()}
        for s, d in rows.items():
            for k, v in d.items():
                glob[s][k] += v
    trans_out["__global__"] = {s: _norm(dict(d)) for s, d in glob.items()}
    return dict(tau=taus, prior=prior, trans=trans_out)


def bl_last_seen(ctx, q, stats):
    return _smoothed_onehot(ctx.locs, q["last_recept"])


def bl_last_known(ctx, q, stats):
    rec, _ = ctx.last_known(q["obj"], q["qt"])
    return _smoothed_onehot(ctx.locs, rec if rec is not None else q["last_recept"])


def bl_hist_freq(ctx, q, stats):
    w = defaultdict(float)
    for t0, t1, rec in ctx.pos_before(q["obj"], q["qt"]):
        w[rec] += (t1 - t0 + 1)
    return _norm({r: w.get(r, 0.0) + 0.5 for r in ctx.locs})


def _mix(ctx, q, stats, dest_dist):
    cls = ctx.obj[q["obj"]]["cls"]
    tau = stats["tau"].get(cls, 1e9)
    rec, t_last = ctx.last_known(q["obj"], q["qt"])
    if rec is None:
        rec, t_last = q["last_recept"], q["last_t"]
    dt = (q["qt"] - t_last) * 5
    p_stay = math.exp(-dt / tau)
    hot = _smoothed_onehot(ctx.locs, rec)
    return {r: p_stay * hot[r] + (1.0 - p_stay) * dest_dist.get(r, 0.0) for r in ctx.locs}


def bl_persistence(ctx, q, stats):
    cls = ctx.obj[q["obj"]]["cls"]
    return _mix(ctx, q, stats, type_dist_to_locs(ctx.home, stats["prior"].get(cls, {})))


def bl_markov(ctx, q, stats):
    cls = ctx.obj[q["obj"]]["cls"]
    rec, _ = ctx.last_known(q["obj"], q["qt"])
    st = ctx.ltype[rec if rec is not None else q["last_recept"]]
    row = stats["trans"].get(cls, {}).get(st) or stats["trans"]["__global__"].get(st)
    if not row:
        return bl_persistence(ctx, q, stats)
    return _mix(ctx, q, stats, type_dist_to_locs(ctx.home, row))


BASELINES = {"last_seen": bl_last_seen, "last_known": bl_last_known,
             "hist_freq": bl_hist_freq, "persistence": bl_persistence,
             "markov": bl_markov}

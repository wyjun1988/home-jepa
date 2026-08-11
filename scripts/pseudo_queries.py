"""Label-free pseudo-queries from target-domain logs (DESIGN 11 / P8).

For consecutive same-class sightings (j -> j+1) in an episode's EVENTS, the
next sighting's surface is a pseudo-label for "where is it" queries at times
in between. No ground truth is read. Reliability: only classes that never
co-appear twice in one glance (singleton-ish, id-free estimate) -- for those,
"next sighting of the class" is almost surely the same instance."""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homejepa.sim import dtbin  # noqa: E402


def singletonish(ep):
    mx = {}
    for g in ep.get("glances", []):
        objs = g["objs"]
        for c in set(objs):
            mx[c] = max(mx.get(c, 0), objs.count(c))
    return {c for c, m in mx.items() if m <= 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.dirname(__file__), "..", "data", "homer_v4", "adapt"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "homer_v4", "pseudo"))
    ap.add_argument("--per-pair", type=int, default=2)
    ap.add_argument("--max-gap", type=int, default=2160,
                    help="ticks; drop pairs whose absence gap exceeds this (3h)")
    ap.add_argument("--room-moves-only", type=int, default=1,
                    help="drop within-room pseudo-moved pairs: a next sighting on "
                         "another surface of the SAME room is usually a double move")
    ap.add_argument("--moved-need-witness", type=int, default=1,
                    help="for pseudo-moved pairs, require witnessed absence at the old surface")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(0)
    n_ep = nq = nmoved = 0
    for fp in sorted(glob.glob(os.path.join(args.src, "ep_*.json"))):
        ep = json.load(open(fp))
        ok_cls = singletonish(ep)
        ocls = {o["id"]: o["cls"] for o in ep["home"]["objects"]}
        runs = {}
        for e in ep["events"]:
            if e["type"] != "POS":
                continue
            c = ocls[e["obj"]]
            if c in ok_cls:
                runs.setdefault(c, []).append((e["t0"], e["t1"], e["recept"], e["room"]))
        # witness index: per receptacle type, glance times where >=2 distinct
        # objects were seen (so a missing target is informative)
        wit = {}
        for g in ep.get("glances", []):
            if len(g["objs"]) >= 2:
                for ft in set(g["furn"]):
                    wit.setdefault(ft, []).append((g["t1"], set(g["objs"])))
        rtype = {rc["id"]: rc["type"] for rc in ep["home"]["recepts"]}

        def witnessed_absence(rec, c, t0, t1):
            for gt_, objs in wit.get(rtype[rec], ()):
                if t0 <= gt_ <= t1 and c not in objs:
                    return True
            return False

        queries = []
        for c, rs in runs.items():
            rs.sort()
            # a stand-in object id of this class (loader only uses its class)
            oid = next(o["id"] for o in ep["home"]["objects"] if o["cls"] == c)
            for (t0a, t1a, ra, rma), (t0b, t1b, rb, rmb) in zip(rs, rs[1:]):
                gap = t0b - t1a
                if gap < 72:                      # < 6 min: not a real absence
                    continue
                if args.max_gap and gap > args.max_gap:
                    continue                      # long gaps hide double-moves
                moved = rb != ra
                if moved and args.room_moves_only and rmb == rma:
                    continue
                if moved and args.moved_need_witness and \
                        not witnessed_absence(ra, c, t1a, t0b):
                    continue
                for _ in range(args.per_pair):
                    qt = rng.randint(t1a + 36, t0b - 1)
                    dt = (qt - t1a) * 5
                    queries.append(dict(
                        qt=qt, obj=oid, gt_recept=rb, gt_room=rmb,
                        last_recept=ra, last_room=rma, last_t=t1a,
                        dt=dt, moved=int(moved), moved_room=int(rmb != rma),
                        mover=1, n_moves=0, dtbin=dtbin(dt), pseudo=1))
        if not queries:
            continue
        ep["queries"] = queries
        ep.pop("gt", None)          # make label-freedom structural
        ep["gt"] = []
        ep.pop("gt_moves", None)
        ep["gt_moves"] = []
        with open(os.path.join(args.out, os.path.basename(fp)), "w") as f:
            json.dump(ep, f)
        n_ep += 1
        nq += len(queries)
        nmoved += sum(q["moved"] for q in queries)
    print("pseudo episodes %d  queries %d  pseudo-moved %.1f%%"
          % (n_ep, nq, 100.0 * nmoved / max(1, nq)))


if __name__ == "__main__":
    main()

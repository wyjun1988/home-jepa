"""Dataset sanity/stat report: moved fraction, dt distribution, per-class move
rates, observation coverage. DESIGN §7-2."""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, "ep_*.json")))
    print("episodes:", len(files))

    nq = moved = 0
    dtb = Counter()
    dtb_room = Counter()
    dtb_moved = Counter()
    mover_kind = Counter()
    cls_moves = Counter()
    cls_count = Counter()
    cls_q = Counter()
    cls_q_moved = Counter()
    ev_counts = Counter()
    rooms_hist = Counter()
    objs_hist = []
    seen_frac = []
    nmoves_all = []

    for fp in files:
        ep = json.load(open(fp))
        obj_cls = {o["id"]: o["cls"] for o in ep["home"]["objects"]}
        rooms_hist[len(ep["home"]["rooms"])] += 1
        objs_hist.append(len(ep["home"]["objects"]))
        for e in ep["events"]:
            ev_counts[e["type"]] += 1
        seen = set(e["obj"] for e in ep["events"] if e["type"] == "POS")
        seen_frac.append(len(seen) / max(1, len(ep["home"]["objects"])))
        for o in ep["home"]["objects"]:
            cls_count[o["cls"]] += 1
        for m in ep["gt_moves"]:
            cls_moves[obj_cls[m["obj"]]] += 1
        nmoves_all.append(len(ep["gt_moves"]) / ep["days"])
        for q in ep["queries"]:
            nq += 1
            moved += q["moved"]
            movedroom = q.get("moved_room", 0)
            dtb_room[q["dtbin"]] += movedroom
            dtb[q["dtbin"]] += 1
            if q["moved"]:
                dtb_moved[q["dtbin"]] += 1
                c = obj_cls[q["obj"]]
                cls_q_moved[c] += 1
                mover_kind["user" if q["mover"] == 0 else "housemate"] += 1
            cls_q[obj_cls[q["obj"]]] += 1

    print("\n== queries ==")
    print("total: %d   moved: %d (%.1f%%)" % (nq, moved, 100.0 * moved / max(1, nq)))
    for b in ["<10m", "10-60m", "1-6h", ">6h"]:
        n = dtb[b]
        print("  dt %-7s n=%-6d moved(recept)=%.1f%%  moved(room)=%.1f%%"
              % (b, n, 100.0 * dtb_moved[b] / max(1, n), 100.0 * dtb_room[b] / max(1, n)))
    print("  moved by: %s" % dict(mover_kind))
    print("\n== world ==")
    print("rooms hist:", dict(sorted(rooms_hist.items())))
    print("objects/home: min %d med %d max %d" % (min(objs_hist),
          sorted(objs_hist)[len(objs_hist) // 2], max(objs_hist)))
    print("gt moves/home/day: %.1f" % (sum(nmoves_all) / len(nmoves_all)))
    print("objects seen>=1 frac: %.2f" % (sum(seen_frac) / len(seen_frac)))
    print("events/ep:", {k: round(v / len(files), 1) for k, v in ev_counts.items()})
    print("\n== per-class (moves/instance/day | query share | moved%% among its queries) ==")
    days = 3
    rows = []
    for c in sorted(cls_count, key=lambda c: -cls_moves[c] / max(1, cls_count[c])):
        mpd = cls_moves[c] / max(1, cls_count[c]) / days
        qs = cls_q[c]
        mv = 100.0 * cls_q_moved[c] / max(1, qs)
        rows.append("  %-13s %5.2f  q=%-5d moved=%.0f%%" % (c, mpd, qs, mv))
    print("\n".join(rows))


if __name__ == "__main__":
    main()

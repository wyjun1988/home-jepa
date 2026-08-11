"""Diagnostic: what fraction of POS events are re-confirmations (object seen
again in the same room as its previous sighting) vs genuine changes?"""
import argparse
import glob
import json
import os
from collections import Counter

TICK_SEC = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "v0", "test"))
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    kind = Counter()
    gaps = []
    per_ep = []
    for fp in sorted(glob.glob(os.path.join(args.dir, "ep_*.json")))[:args.limit]:
        ep = json.load(open(fp))
        last = {}
        n_pos = 0
        for e in sorted(ep["events"], key=lambda x: x["t1"]):
            if e["type"] != "POS":
                continue
            n_pos += 1
            oid, room = e["obj"], e["room"]
            if oid not in last:
                kind["first_sighting"] += 1
            elif last[oid][0] == room:
                kind["reconfirm_same_room"] += 1
                gaps.append((e["t0"] - last[oid][1]) * TICK_SEC / 60.0)
            else:
                kind["change_new_room"] += 1
            last[oid] = (room, e["t1"])
        per_ep.append(n_pos)

    tot = sum(kind.values())
    print("POS events: %.0f per episode (%d analysed)" % (sum(per_ep) / len(per_ep), tot))
    for k, v in kind.most_common():
        print("  %-20s %6d  %5.1f%%" % (k, v, 100.0 * v / tot))
    if gaps:
        gaps.sort()
        print("\nre-confirmation gap (min since previous sighting of same object):")
        print("  median %.1f   p25 %.1f   p75 %.1f   p90 %.1f"
              % (gaps[len(gaps) // 2], gaps[len(gaps) // 4],
                 gaps[3 * len(gaps) // 4], gaps[int(0.9 * len(gaps))]))
        print("  under 30 min: %.1f%%" % (100.0 * sum(1 for g in gaps if g < 30) / len(gaps)))


if __name__ == "__main__":
    main()

"""Pseudo-label quality audit (P8 verification).

The adaptation pipeline builds pseudo-labels WITHOUT ground truth. This script
is the one place we open GT -- purely to measure how noisy those pseudo-labels
are. It never feeds anything back into training.

Measures, on the SOURCE episodes (which still carry gt):
  - agreement: pseudo gt_recept == true receptacle at qt
  - agreement by stratum (pseudo says moved / stayed)
  - the failure mode: pseudo-label points at the next sighting's surface, but
    the object was somewhere else at qt (it moved twice, or the next sighting
    is a different instance of the same class)
"""
import argparse
import glob
import json
import os
import sys
from bisect import bisect_right
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from pseudo_queries import singletonish   # noqa: E402


def true_loc(moves, t):
    i = bisect_right(moves, [t - 1, 10 ** 9]) - 1
    return moves[max(0, i)][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.dirname(__file__), "..", "data", "homer_v4", "adapt"))
    ap.add_argument("--pseudo", default=os.path.join(os.path.dirname(__file__), "..", "data", "homer_v4", "pseudo"))
    args = ap.parse_args()

    agree = Counter()
    tot = Counter()
    room_agree = Counter()
    n_cls_multi = 0
    n_ep = 0
    for pf in sorted(glob.glob(os.path.join(args.pseudo, "ep_*.json"))):
        sf = os.path.join(args.src, os.path.basename(pf))
        if not os.path.exists(sf):
            continue
        src = json.load(open(sf))
        ps = json.load(open(pf))
        gt = {g["obj"]: g["moves"] for g in src["gt"]}
        lroom = {rc["id"]: rc["room"] for rc in src["home"]["recepts"]}
        ocls = {o["id"]: o["cls"] for o in src["home"]["objects"]}
        # how many objects share the class we accepted as "singleton-ish"?
        keep = singletonish(src)
        cls_count = Counter(ocls.values())
        n_cls_multi += sum(1 for c in keep if cls_count[c] > 1)
        n_ep += 1
        for q in ps["queries"]:
            oid = q["obj"]
            if oid not in gt:
                continue
            truth = true_loc(gt[oid], q["qt"])
            stratum = "pseudo_moved" if q["moved"] else "pseudo_stayed"
            for k in ("all", stratum):
                tot[k] += 1
                agree[k] += int(truth == q["gt_recept"])
                room_agree[k] += int(lroom.get(truth, -1) == q["gt_room"])
    print("episodes checked: %d" % n_ep)
    print("accepted classes that actually have >1 instance: %.1f per episode"
          % (n_cls_multi / max(1, n_ep)))
    for k in ("all", "pseudo_moved", "pseudo_stayed"):
        if tot[k]:
            print("%-14s n=%-6d  surface-agreement %.3f   room-agreement %.3f"
                  % (k, tot[k], agree[k] / tot[k], room_agree[k] / tot[k]))


if __name__ == "__main__":
    main()

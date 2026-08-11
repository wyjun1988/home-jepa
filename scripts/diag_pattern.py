"""Is there learnable pattern in where moved objects end up, and what do
chance baselines actually score? Answers: (a) rooms-per-home, so what "random"
means; (b) simple informed guesses (home room, modal room, class prior);
(c) a destination-concentration ceiling proxy fitted on train GT."""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "v0")


def room_type(ep, rid):
    return ep["home"]["rooms"][rid]["type"]


def main():
    # ---- fit destination priors on train GT ----
    dst_by_cls = defaultdict(Counter)              # class -> dst type
    dst_by_cls_src = defaultdict(Counter)          # (class, src type) -> dst type
    dst_by_cls_hour = defaultdict(Counter)         # (class, hour bucket) -> dst type
    for fp in sorted(glob.glob(os.path.join(DATA, "train", "ep_*.json")))[:300]:
        ep = json.load(open(fp))
        cls = {o["id"]: o["cls"] for o in ep["home"]["objects"]}
        for m in ep["gt_moves"]:
            if m["src"] < 0 or m["dst"] < 0:
                continue
            c = cls[m["obj"]]
            st, dt_ = room_type(ep, m["src"]), room_type(ep, m["dst"])
            hour = (m["t"] % 17280) // (17280 // 4)      # 4 buckets/day
            dst_by_cls[c][dt_] += 1
            dst_by_cls_src[(c, st)][dt_] += 1
            dst_by_cls_hour[(c, hour)][dt_] += 1

    # ---- evaluate on test moved queries ----
    nrooms = Counter()
    n = 0
    hit = Counter()
    hit_hm = Counter()
    n_hm = 0
    top1_mass = []
    for fp in sorted(glob.glob(os.path.join(DATA, "test", "ep_*.json"))):
        ep = json.load(open(fp))
        obj = {o["id"]: o for o in ep["home"]["objects"]}
        nrooms[len(ep["home"]["rooms"])] += 1
        # modal observed room per object (from POS events up to each query is
        # overkill; whole-episode modal is an optimistic version)
        seen = defaultdict(Counter)
        for e in ep["events"]:
            if e["type"] == "POS":
                seen[e["obj"]][e["room"]] += e["t1"] - e["t0"] + 1
        for q in ep["queries"]:
            if not q["moved"]:
                continue
            n += 1
            hm = q.get("mover") is not None and q["mover"] != 0
            n_hm += hm
            R = len(ep["home"]["rooms"])
            gt = q["gt_room"]
            gtt = room_type(ep, gt)
            o = obj[q["obj"]]

            def score(name, ok):
                hit[name] += ok
                if hm:
                    hit_hm[name] += ok

            score("uniform", 1.0 / R)
            score("uniform_excl_anchor", 1.0 / max(1, R - 1))
            score("home_room", int(gt == o["home_room"]))
            sc = seen.get(q["obj"])
            score("modal_seen_room", int(sc and gt == sc.most_common(1)[0][0]))
            c = o["cls"]
            st = room_type(ep, q["last_room"])
            hour = (q["qt"] % 17280) // (17280 // 4)
            for name, table in [("prior_cls", dst_by_cls.get(c)),
                                ("prior_cls_src", dst_by_cls_src.get((c, st))),
                                ("prior_cls_hour", dst_by_cls_hour.get((c, hour)))]:
                if table:
                    guess = table.most_common(1)[0][0]
                    score(name, int(gtt == guess))
                    tot = sum(table.values())
                    if name == "prior_cls_src":
                        top1_mass.append(table.most_common(1)[0][1] / tot)
                else:
                    score(name, 0)

    print("test homes rooms histogram:", dict(sorted(nrooms.items())))
    print("moved queries n=%d  (housemate-moved n=%d)\n" % (n, n_hm))
    print("%-22s %-10s %-14s" % ("guess", "moved all", "housemate-moved"))
    for k in ["uniform", "uniform_excl_anchor", "home_room", "modal_seen_room",
              "prior_cls", "prior_cls_src", "prior_cls_hour"]:
        print("%-22s %-10.3f %-14.3f" % (k, hit[k] / n, hit_hm[k] / max(1, n_hm)))
    top1_mass.sort()
    m = len(top1_mass)
    print("\ntrain-GT destination concentration P(modal dst-type | class, src-type):")
    print("  median %.2f   p25 %.2f   p75 %.2f  -> destinations are far from uniform"
          % (top1_mass[m // 2], top1_mass[m // 4], top1_mass[3 * m // 4]))


if __name__ == "__main__":
    main()

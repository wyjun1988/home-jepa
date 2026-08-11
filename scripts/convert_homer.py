import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homejepa.homer import convert_day    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.dirname(__file__), "..", "data", "homer_plus"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "homer"))
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--user-seeds", type=int, default=3,
                    help="observer schedules per day (observation diversity)")
    args = ap.parse_args()

    outdir = os.path.join(args.out, "test" if args.split == "test" else "adapt")
    os.makedirs(outdir, exist_ok=True)
    n = nq = nmoved = 0
    for hh_i, hh in enumerate(["HouseholdA", "HouseholdB", "HouseholdC"]):
        for fp in sorted(glob.glob(os.path.join(args.src, hh, "routines_%s" % args.split, "*.json"))):
            day = json.load(open(fp))
            dayname = os.path.splitext(os.path.basename(fp))[0]
            for us in range(args.user_seeds):
                seed = 900000 + hh_i * 10000 + int(dayname) * 10 + us
                ep = convert_day(day, seed, ep_id=seed)
                with open(os.path.join(outdir, "ep_%06d.json" % seed), "w") as f:
                    json.dump(ep, f)
                n += 1
                nq += len(ep["queries"])
                nmoved += sum(q["moved"] for q in ep["queries"])
        print("%s done (%d eps so far)" % (hh, n), flush=True)
    print("episodes %d  queries %d  moved %.1f%%" % (n, nq, 100.0 * nmoved / max(1, nq)))


if __name__ == "__main__":
    main()

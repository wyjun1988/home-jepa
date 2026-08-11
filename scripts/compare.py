"""Merged comparison table over every result file present in results/."""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homejepa.metrics import ROW_ORDER   # noqa: E402

METRICS = [("top1", "%.3f"), ("top2", "%.3f"), ("nll", "%.2f"), ("search", "%.2f")]
KEY_ROWS = ["all", "stayed", "moved", "moved_by:housemate", "moved_by:user",
            "dt:>6h", "moved|dt:>6h"]


def load_gens(resdir):
    fp = os.path.join(resdir, "GENERATIONS.json")
    return json.load(open(fp)) if os.path.exists(fp) else {}


def load_all(resdir, split="test"):
    out = {}
    bl_path = os.path.join(resdir, "baselines_%s.json" % split)
    if os.path.exists(bl_path):
        out.update(json.load(open(bl_path)))
    for fp in sorted(glob.glob(os.path.join(resdir, "*_%s.json" % split))):
        name = os.path.basename(fp)[:-len("_%s.json" % split)]
        if name == "baselines":
            continue
        out[name.replace("supervised_", "")] = json.load(open(fp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--rows", default="key", choices=["key", "all"])
    ap.add_argument("--only", default=None, help="comma-separated subset of columns")
    ap.add_argument("--gen", default="latest",
                    help="data generation to show (see results/GENERATIONS.json); "
                         "'all' shows everything with a mixing warning")
    args = ap.parse_args()

    data = load_all(args.results, args.split)
    gens = load_gens(args.results)

    def gen_of(name):
        for cand in (name, "supervised_" + name, name + "_" + args.split,
                     "supervised_%s_%s" % (name, args.split), "baselines_" + args.split):
            if cand in gens:
                return gens[cand]["gen"]
        return "unknown"

    if args.only:
        want = args.only.split(",")
        data = {k: v for k, v in data.items() if k in want}
    elif args.gen != "all":
        # results from different data generations are NOT comparable; default to
        # the newest generation present unless the user explicitly asks for one
        present = {gen_of(k) for k in data}
        target = args.gen if args.gen != "latest" else \
            sorted(present, key=lambda g: (g.startswith("unknown"), g))[-1]
        data = {k: v for k, v in data.items() if gen_of(k) == target}
        print("[generation filter] %s  (%d columns; --gen all 로 해제)"
              % (target, len(data)))
    if args.only or args.gen == "all":
        mixed = sorted({gen_of(k) for k in data})
        if len(mixed) > 1:
            print("!! 경고: 서로 다른 데이터 세대를 한 표에 비교 중 —", ", ".join(mixed))
    if not data:
        print("no results found in %s" % args.results)
        return
    cols = list(data)
    rows = KEY_ROWS if args.rows == "key" else ROW_ORDER
    ref = data[cols[0]]

    for metric, fmt in METRICS:
        hi = metric in ("top1", "top2", "top3")
        print("\n### %s%s" % (metric.upper(), "" if hi else "  (lower is better)"))
        head = "%-24s" % "bucket" + "".join("%-14s" % c[:13] for c in cols)
        print(head)
        print("-" * len(head))
        for b in rows:
            if b not in ref:
                continue
            vals = [data[c].get(b, {}).get(metric) for c in cols]
            present = [v for v in vals if v is not None]
            if not present:
                continue
            best = (max if hi else min)(present)
            cells = ""
            for v in vals:
                s = (fmt % v) if v is not None else "-"
                cells += "%-14s" % (s + (" *" if v == best else ""))
            print("%-24s" % ("%s(%d)" % (b, ref[b]["n"])) + cells)


if __name__ == "__main__":
    main()

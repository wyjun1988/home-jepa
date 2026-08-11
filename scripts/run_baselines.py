import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homejepa.baselines import BASELINES, EpisodeCtx, fit_stats     # noqa: E402
from homejepa.metrics import Aggregator, format_table, score_query  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "v0"))
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    args = ap.parse_args()

    train_files = sorted(glob.glob(os.path.join(args.data, "train", "ep_*.json")))
    eval_files = sorted(glob.glob(os.path.join(args.data, args.eval_split, "ep_*.json")))
    print("fit on %d train eps; eval on %d %s eps" % (len(train_files), len(eval_files), args.eval_split))
    stats_path = os.path.join(args.out, "baseline_stats.json")
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(stats_path):
        stats = json.load(open(stats_path))
    else:
        stats = fit_stats(train_files)
        json.dump(stats, open(stats_path, "w"))
    print("tau (hours, sample):", {c: round(stats["tau"][c] / 3600, 1)
          for c in ["phone", "keys", "laptop", "book", "remote", "plant"] if c in stats["tau"]})

    aggs = {name: Aggregator() for name in BASELINES}
    for fp in eval_files:
        ep = json.load(open(fp))
        ctx = EpisodeCtx(ep)
        for q in ep["queries"]:
            for name, fn in BASELINES.items():
                prob = fn(ctx, q, stats)
                aggs[name].add(q, score_query(prob, q["gt_recept"], len(ctx.locs),
                                              room_of=ctx.lroom, gt_room=q["gt_room"]))
    summaries = {n: a.summary() for n, a in aggs.items()}
    print()
    print(format_table(summaries))
    json.dump(summaries, open(os.path.join(args.out, "baselines_%s.json" % args.eval_split), "w"), indent=1)
    print("\nsaved -> results/baselines_%s.json" % args.eval_split)


if __name__ == "__main__":
    main()

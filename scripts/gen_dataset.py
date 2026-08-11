import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homejepa.world import sample_home                     # noqa: E402
from homejepa.sim import simulate_episode                  # noqa: E402

SPLIT_SEED_BASE = {"train": 100000, "val": 200000, "test": 300000,
                   "pretrain": 500000}   # unlabeled pool for JEPA scaling; must not collide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "v0"))
    ap.add_argument("--split", required=True, choices=list(SPLIT_SEED_BASE))
    ap.add_argument("--homes", type=int, required=True)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--queries", type=int, default=120)
    ap.add_argument("--packs", default=None,
                    help="LLM-authored household packs json; cycled across homes")
    args = ap.parse_args()
    packs = json.load(open(args.packs)) if args.packs else None

    outdir = os.path.join(args.out, args.split)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    for i in range(args.homes):
        seed = SPLIT_SEED_BASE[args.split] + i
        rng = random.Random(seed)
        if packs:
            from homejepa.routines import apply_pack
            pk = packs[i % len(packs)]
            home = sample_home(rng, home_id=seed, n_agents=len(pk["personas"]))
            apply_pack(home, pk, rng)
        else:
            home = sample_home(rng, home_id=seed)
        ep = simulate_episode(home, args.days, rng, args.queries)
        ep["seed"] = seed
        if packs:
            ep["pack"] = pk.get("household", str(i % len(packs)))
        with open(os.path.join(outdir, "ep_%06d.json" % seed), "w") as f:
            json.dump(ep, f)
        if (i + 1) % 20 == 0 or i == args.homes - 1:
            print("[%s] %d/%d  (%.1fs)" % (args.split, i + 1, args.homes, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

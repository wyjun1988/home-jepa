"""ref-past queries: the first direct measurement of ASSOCIATION ability.

CONCEPT_REVIEW 1.2: the standard strata cannot measure "it vanished here and
appeared there => same object moved", because an observed move updates the
reference sighting and reclassifies the query as stayed. Here we pin the
reference track to a PAST sighting on purpose ("the laptop I saw on the desk
this morning -- where is it now?"): a later re-sighting elsewhere IS in the
log, and only a model that associates it with the reference can answer.

Generation is GT-side (privileged, like sim queries); the model still sees an
id-free log in noid mode. No data files are mutated -- queries are built in
memory on top of the existing test episodes.
"""
import argparse
import glob
import json
import os
import random
import sys
from bisect import bisect_right
from collections import Counter, defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from homejepa.model import EpTensors, make_batch          # noqa: E402
from homejepa.sim import dtbin                            # noqa: E402
from reeval import build_supervised, build_jepa_probe     # noqa: E402


def true_loc(moves, t):
    i = bisect_right(moves, [t - 1, 10 ** 9]) - 1
    return moves[max(0, i)][1]


def build_refpast(ep, rng, per_ep=60):
    lroom = {rc["id"]: rc["room"] for rc in ep["home"]["recepts"]}
    gt = {g["obj"]: g["moves"] for g in ep["gt"]}
    runs = defaultdict(list)
    for e in ep["events"]:
        if e["type"] == "POS":
            runs[e["obj"]].append((e["t0"], e["t1"], e["recept"]))
    vis = {o: rs for o, rs in runs.items()}
    T = ep["days"] * 17280
    out = []
    for o, rs in runs.items():
        rs.sort()
        for (a0, a1, ra), (b0, b1, rb) in zip(rs, rs[1:]):
            if rb == ra:
                continue
            lo, hi = b1 + 36, min(b1 + 1440, T - 1)
            if hi <= lo:
                continue
            for _ in range(2):
                qt = rng.randint(lo, hi)
                if any(v0 <= qt <= v1 + 1 for v0, v1, _ in vis[o]):
                    continue
                cur = true_loc(gt[o], qt)
                if cur not in lroom:          # OUT of home at qt
                    continue
                movers = [m["mover"] for m in ep["gt_moves"]
                          if m["obj"] == o and a1 < m["t"] <= qt]
                dt = (qt - a1) * 5
                out.append(dict(
                    qt=qt, obj=o, gt_recept=cur, gt_room=lroom[cur],
                    last_recept=ra, last_room=lroom[ra], last_t=a1,
                    dt=dt, moved=int(cur != ra), moved_room=int(lroom[cur] != lroom[ra]),
                    mover=(movers[-1] if movers else None), n_moves=len(movers),
                    dtbin=dtbin(dt), refpast=1, resight=rb))
                break
    rng.shuffle(out)
    return out[:per_ep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "v5", "test"))
    ap.add_argument("--ckpts", default="supervised_two_head_v5,jepa_jepa_v5_ft_fair_probe,jepa_jepa_v5_scr_fair_probe")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = ap.parse_args()
    dev = torch.device(args.device)
    rng = random.Random(0)

    eps = []
    ref_stats = Counter()
    for fp in sorted(glob.glob(os.path.join(args.data, "ep_*.json"))):
        ep = json.load(open(fp))
        qs = build_refpast(ep, rng)
        if not qs:
            continue
        for q in qs:
            ref_stats["n"] += 1
            ref_stats["assoc_easy(gt=재목격지)"] += int(q["gt_recept"] == q["resight"])
            ref_stats["moved(vs 참조)"] += int(q["moved"])
        ep = dict(ep, queries=qs)
        eps.append(EpTensors(ep, 256, noid=True))
    n = ref_stats["n"]
    print("ref-past 질의 %d개 | 연관하면 정답(gt=재목격지) %.2f | moved(참조 기준) %.2f"
          % (n, ref_stats["assoc_easy(gt=재목격지)"] / n, ref_stats["moved(vs 참조)"] / n))

    # non-learned references
    ref = {"참조트랙(과거앵커)": Counter(), "완벽연관(재목격지 답)": Counter()}
    for ep in eps:
        for q in ep.queries:
            m = q["meta"]
            gt = q["gt"]
            ref["참조트랙(과거앵커)"]["hit"] += int(ep.loc_pos[m["last_recept"]] == gt)
            ref["완벽연관(재목격지 답)"]["hit"] += int(ep.loc_pos[m["resight"]] == gt)
            for r in ref.values():
                r["n"] += 0
    tot = sum(len(ep.queries) for ep in eps)
    for name, c in ref.items():
        print("%-24s top-1 %.3f" % (name, c["hit"] / tot))

    results = {}
    for name in args.ckpts.split(","):
        fp = os.path.join(os.path.dirname(__file__), "..", "results", name + ".pt")
        if not os.path.exists(fp):
            print("skip %s (no ckpt)" % name)
            continue
        ck = torch.load(fp, map_location=dev, weights_only=False)
        model, _ = (build_jepa_probe if name.startswith("jepa_") else build_supervised)(ck, dev)
        model = model.to(dev).eval()
        samples = [(e, q) for e in range(len(eps)) for q in range(len(eps[e].queries))]
        agg = defaultdict(Counter)
        with torch.no_grad():
            for i in range(0, len(samples), 256):
                chunk = samples[i:i + 256]
                b = make_batch(eps, chunk, dev)
                p = model.log_prob(b).exp().cpu()
                for j, (ei, qi) in enumerate(chunk):
                    ep = eps[ei]
                    q = ep.queries[qi]
                    m = q["meta"]
                    pv = p[j, :ep.n_loc]
                    order = pv.argsort(descending=True)
                    keys = ["all",
                            "assoc(gt=재목격지)" if q["gt"] == ep.loc_pos[m["resight"]] else "further_moved",
                            "cls_multi" if m.get("multi") else "cls_single"]
                    for k in keys:
                        a = agg[k]
                        a["n"] += 1
                        a["t1"] += int(order[0].item() == q["gt"])
                        a["t2"] += int(q["gt"] in order[:2].tolist())
                        a["hit_resight"] += int(order[0].item() == ep.loc_pos[m["resight"]])
        print("\n=== %s ===" % name)
        print("%-22s %6s %7s %7s %12s" % ("층", "n", "top1", "top2", "1순위=재목격지"))
        res = {}
        for k in ("all", "assoc(gt=재목격지)", "further_moved", "cls_single", "cls_multi"):
            a = agg[k]
            if not a["n"]:
                continue
            row = dict(n=a["n"], top1=a["t1"] / a["n"], top2=a["t2"] / a["n"],
                       resight=a["hit_resight"] / a["n"])
            res[k] = row
            print("%-22s %6d %7.3f %7.3f %12.3f"
                  % (k, a["n"], row["top1"], row["top2"], row["resight"]))
        results[name] = res
        del model
    if args.tag:
        json.dump(results, open(os.path.join(os.path.dirname(__file__), "..", "results",
                                             "%s.json" % args.tag), "w"), indent=1)


if __name__ == "__main__":
    main()

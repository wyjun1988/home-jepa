"""Zero-shot transfer evaluation: sim-trained baselines-stats and model
checkpoints, evaluated unchanged on a converted external dataset (HOMER+)."""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homejepa.baselines import BASELINES, EpisodeCtx                # noqa: E402
from homejepa.metrics import Aggregator, score_query                # noqa: E402
from homejepa.model import load_split                               # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from reeval import build_supervised, build_jepa_probe               # noqa: E402
from train_supervised import evaluate                               # noqa: E402


def log_tau(ep, lo=600.0, hi=90 * 86400.0):
    """Label-free stay-time scale from the observation log alone: consecutive
    sightings of the same object are (gap, same-room?) samples -- no GT used.
    MLE for an exponential survival on those pairs, grid over [lo, hi] sec."""
    import math
    pairs = []
    for oid, js in ep.obj_know.items():
        for a, b in zip(js, js[1:]):
            gap = (ep.t1[b] - ep.t1[a]) * 5.0
            if gap >= 20:
                pairs.append((gap, int(ep.ev_rec[a] == ep.ev_rec[b])))
    if not pairs:
        return hi
    best, best_ll = hi, -1e18
    tau = lo
    while tau <= hi:
        ll = 0.0
        for g, same in pairs:
            ps = math.exp(-g / tau)
            ll += math.log(max(1e-9, ps if same else 1.0 - ps))
        if ll > best_ll:
            best_ll, best = ll, tau
        tau *= 1.45
    return best


def eval_recalibrated(model, eps, taus, dev, batch=256):
    """Replace the model's implied move-rate with the log-estimated one, keep
    its conditional destination: p = p_stay*onehot(anchor) + (1-p_stay)*p_off."""
    import math
    import torch.nn.functional as F
    from homejepa.model import make_batch
    agg = Aggregator()
    samples = [(e, q) for e in range(len(eps)) for q in range(len(eps[e].queries))]
    model.eval()
    with torch.no_grad():
        for i in range(0, len(samples), batch):
            chunk = samples[i:i + batch]
            b = make_batch(eps, chunk, dev)
            p = model.log_prob(b).exp().cpu()
            for j, (ei, qi) in enumerate(chunk):
                ep = eps[ei]
                q = ep.queries[qi]
                anchor = q["anchor"]
                dt = (q["qt"] - q["anchor_t"]) * 5.0
                p_stay = math.exp(-dt / taus[ei])
                pv = p[j, :ep.n_loc].clone()
                off = pv.clone()
                off[anchor] = 0.0
                off = off / max(1e-9, float(off.sum()))
                pv = (1.0 - p_stay) * off
                pv[anchor] = p_stay
                rid = [rc["id"] for rc in ep.recepts]
                agg.add(q["meta"], score_query({r: float(pv[kk]) for kk, r in enumerate(rid)},
                                               q["meta"]["gt_recept"], len(rid),
                                               room_of=ep.loc_room, gt_room=q["meta"]["gt_room"]))
    model.train()
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "homer", "test"))
    ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--models", default="supervised_flat_v2,jepa_jepa_v3_ft_probe,"
                                        "jepa_jepa_5x_ft_probe,jepa_jepa_scratch_probe")
    ap.add_argument("--tag", default="transfer_homer")
    ap.add_argument("--stats", default="baseline_stats_v1.json")
    ap.add_argument("--noid", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = ap.parse_args()
    dev = torch.device(args.device)
    files = sorted(glob.glob(os.path.join(args.data, "ep_*.json")))
    print("eval on %d episodes" % len(files))
    out = {}

    stats = json.load(open(os.path.join(args.results, args.stats)))
    aggs = {n: Aggregator() for n in BASELINES}
    for fp in files:
        ep = json.load(open(fp))
        ctx = EpisodeCtx(ep)
        for q in ep["queries"]:
            for name, fn in BASELINES.items():
                aggs[name].add(q, score_query(fn(ctx, q, stats), q["gt_recept"], len(ctx.locs),
                                              room_of=ctx.lroom, gt_room=q["gt_room"]))
    for n, a in aggs.items():
        out[n] = a.summary()

    eps = load_split(files, 256, noid=args.noid)
    taus = [log_tau(ep) for ep in eps]
    print("label-free tau from logs: median %.1f h" %
          (sorted(taus)[len(taus) // 2] / 3600.0))
    for name in args.models.split(","):
        fp = os.path.join(args.results, name + ".pt")
        ck = torch.load(fp, map_location=dev)
        model, _ = (build_jepa_probe if name.startswith("jepa_") else build_supervised)(ck, dev)
        model = model.to(dev)
        nll, agg = evaluate(model, eps, dev)
        short = name.replace("supervised_", "").replace("jepa_jepa_", "jepa_").replace("_probe", "")
        out[short] = agg.summary()
        out[short + "+recal"] = eval_recalibrated(model, eps, taus, dev).summary()
        print("%-22s zero-shot t1 %.3f | +recal t1 %.3f"
              % (short, out[short]["all"]["top1"], out[short + "+recal"]["all"]["top1"]), flush=True)
        del model

    json.dump(out, open(os.path.join(args.results, "%s.json" % args.tag), "w"), indent=1)
    print("saved -> results/%s.json" % args.tag)


if __name__ == "__main__":
    main()

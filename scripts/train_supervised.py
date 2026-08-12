import argparse
import glob
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homejepa.model import (EventTransformer, TwoHeadEventTransformer,        # noqa: E402
                            load_split, make_batch)
from homejepa.metrics import Aggregator, format_table, score_query    # noqa: E402

MODELS = {"flat": EventTransformer, "two_head": TwoHeadEventTransformer}


def evaluate(model, eps, device, agg=None, batch=256):
    model.eval()
    samples = [(e, q) for e in range(len(eps)) for q in range(len(eps[e].queries))]
    agg = agg or Aggregator()
    nll_sum = n = 0
    with torch.no_grad():
        for i in range(0, len(samples), batch):
            chunk = samples[i:i + batch]
            b = make_batch(eps, chunk, device)
            logp = model.log_prob(b)
            nll_sum += F.nll_loss(logp, b["gt"], reduction="sum").item()
            n += len(chunk)
            probs = logp.exp().cpu().numpy()
            for j, (ei, qi) in enumerate(chunk):
                ep = eps[ei]
                q = ep.queries[qi]
                loc_ids = [rc["id"] for rc in ep.recepts]
                prob = {lid: float(probs[j, k]) for k, lid in enumerate(loc_ids)}
                agg.add(q["meta"], score_query(prob, q["meta"]["gt_recept"], len(loc_ids),
                                               room_of=ep.loc_room, gt_room=q["meta"]["gt_room"]))
    model.train()
    return nll_sum / max(1, n), agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "v0"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--test-data", default=None,
                    help="evaluate on this dir's test split instead (scale runs: "
                         "train big, test on the standard v5 test)")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--max-events", type=int, default=256)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--model", default="two_head", choices=list(MODELS))
    ap.add_argument("--aux-weight", type=float, default=0.5,
                    help="two_head only: weight of the auxiliary flat-softmax task")
    ap.add_argument("--room-feats", action="store_true",
                    help="add per-room negative-evidence attributes")
    ap.add_argument("--tag", default=None, help="suffix for output files")
    ap.add_argument("--gate-weight", type=float, default=0.0,
                    help="auxiliary BCE on the gate (P(moved)); sharpens the "
                         "absence signal that the mixture loss alone under-uses")
    ap.add_argument("--noid", action="store_true",
                    help="strip cross-track instance identity (DESIGN 11.1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--threads", type=int, default=0,
                    help="cap CPU threads so the machine stays usable (0 = torch default)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device(args.device)
    t0 = time.time()
    tr = load_split(sorted(glob.glob(os.path.join(args.data, "train", "ep_*.json"))), args.max_events, noid=args.noid)
    va = load_split(sorted(glob.glob(os.path.join(args.data, "val", "ep_*.json"))), args.max_events, noid=args.noid)
    te = load_split(sorted(glob.glob(os.path.join(args.test_data or os.path.join(args.data, "test"), "ep_*.json"))), args.max_events, noid=args.noid)
    print("loaded %d/%d/%d eps (%.1fs)" % (len(tr), len(va), len(te), time.time() - t0), flush=True)

    tag = args.tag or args.model
    model = MODELS[args.model](d=args.d, layers=args.layers, max_pos=args.max_events + 2,
                               room_feats=args.room_feats).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print("model %s  params: %.2fM  device: %s" % (args.model, nparam / 1e6, dev), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = 300

    def lr_at(s):
        if s < warm:
            return args.lr * (s + 1) / warm
        p = (s - warm) / max(1, args.steps - warm)
        return 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + torch.cos(torch.tensor(p * 3.141592)).item())

    pool = [(e, q) for e in range(len(tr)) for q in range(len(tr[e].queries))]
    print("train queries:", len(pool), flush=True)
    best_nll, best_state, t0 = 1e9, None, time.time()
    losses = []
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = make_batch(tr, rng.sample(pool, args.bs), dev)
        if args.gate_weight > 0 and hasattr(model, "all_heads"):
            lp, aux, g_logit = model.all_heads(batch)     # single encoder pass
            loss = F.nll_loss(lp, batch["gt"])
            if args.aux_weight > 0:
                loss = loss + args.aux_weight * F.nll_loss(aux, batch["gt"])
            y = (batch["gt"] != batch["anchor"]).float()
            loss = loss + args.gate_weight * F.binary_cross_entropy_with_logits(g_logit, y)
        elif args.aux_weight > 0 and hasattr(model, "both"):
            lp, aux = model.both(batch)
            loss = F.nll_loss(lp, batch["gt"]) + args.aux_weight * F.nll_loss(aux, batch["gt"])
        else:
            loss = F.nll_loss(model.log_prob(batch), batch["gt"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if (step + 1) % args.val_every == 0 or step == args.steps - 1:
            vnll, _ = evaluate(model, va, dev)
            flag = ""
            if vnll < best_nll:
                best_nll = vnll
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                flag = "  *best"
                torch.save(dict(state=best_state, args=vars(args), step=step + 1, val_nll=vnll),
                           os.path.join(args.out, "supervised_%s.pt" % tag))
            print("step %5d  loss %.4f  val_nll %.4f  (%.0fs)%s" %
                  (step + 1, sum(losses) / len(losses), vnll, time.time() - t0, flag), flush=True)
            losses = []

    model.load_state_dict(best_state)
    os.makedirs(args.out, exist_ok=True)
    tnll, agg = evaluate(model, te, dev)
    summ = agg.summary()
    print("\ntest NLL %.4f" % tnll)
    print(format_table({tag: summ}))
    json.dump(summ, open(os.path.join(args.out, "supervised_%s_test.json" % tag), "w"), indent=1)
    print("\nsaved -> results/supervised_%s_test.json, results/supervised_%s.pt" % (tag, tag))


if __name__ == "__main__":
    main()

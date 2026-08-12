"""Track J: stage 1 JEPA pretraining (no labels) -> stage 2 frozen-trunk probe."""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homejepa.jepa import HomeJepa, JepaProbe                     # noqa: E402
from homejepa.model import load_split, make_batch                 # noqa: E402
from homejepa.metrics import Aggregator, format_table, score_query  # noqa: E402


def evaluate(model, eps, device, batch=256):
    model.eval()
    samples = [(e, q) for e in range(len(eps)) for q in range(len(eps[e].queries))]
    agg = Aggregator()
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
                lid = [rc["id"] for rc in ep.recepts]
                agg.add(q["meta"], score_query({r: float(probs[j, k]) for k, r in enumerate(lid)},
                                               q["meta"]["gt_recept"], len(lid),
                                               room_of=ep.loc_room, gt_room=q["meta"]["gt_room"]))
    model.train()
    return nll_sum / max(1, n), agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "v0"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--tag", default="jepa")
    ap.add_argument("--s1-steps", type=int, default=3000)
    ap.add_argument("--s2-steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--max-events", type=int, default=256)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--w-var", type=float, default=1.0)
    ap.add_argument("--w-cov", type=float, default=0.01)
    ap.add_argument("--w-anchor", type=float, default=1.0)
    ap.add_argument("--no-freeze", action="store_true", help="stage 2 finetunes the trunk")
    ap.add_argument("--load-s1", default=None,
                    help="reuse a stage-1 checkpoint instead of pretraining again")
    ap.add_argument("--continue-s1", type=int, default=0,
                    help="with --load-s1: continue stage-1 for N steps (label-free adaptation)")
    ap.add_argument("--s1-data", default=None,
                    help="episode dir for stage-1 sampling (default: --data). For adaptation corpora")
    ap.add_argument("--test-data", default=None,
                    help="episode dir for the final test eval (default: --data test split)")
    ap.add_argument("--scratch", action="store_true",
                    help="skip stage 1 entirely: random init, for the pretraining ablation")
    ap.add_argument("--room-feats", action="store_true")
    ap.add_argument("--event-horizons", action="store_true",
                    help="stage 1 predicts the k-th next sighting, k in {1,2,4}")
    ap.add_argument("--aug-gap", type=float, default=0.0,
                    help="structured masking: prob of a suffix-truncated hard copy per query")
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--align-teacher", action="store_true",
                    help="s1: use only (query,k) pairs whose k-th reveal matches the "
                         "CURRENT state at qt (privileged; CONCEPT_REVIEW 2.1 experiment)")
    ap.add_argument("--aux-weight", type=float, default=0.5,
                    help="stage-2 auxiliary flat-softmax task (Track S parity)")
    ap.add_argument("--gate-weight", type=float, default=0.0,
                    help="stage-2 gate BCE supervision (Track S parity)")
    ap.add_argument("--noid", action="store_true",
                    help="strip cross-track instance identity (DESIGN 11.1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    tr = load_split(sorted(glob.glob(os.path.join(args.data, "train", "ep_*.json"))),
                    args.max_events, aug_gap=args.aug_gap, noid=args.noid)
    va = load_split(sorted(glob.glob(os.path.join(args.data, "val", "ep_*.json"))), args.max_events, noid=args.noid)
    te_dir = args.test_data if args.test_data else os.path.join(args.data, "test")
    te = load_split(sorted(glob.glob(os.path.join(te_dir, "ep_*.json"))), args.max_events, noid=args.noid)
    tr_s1 = tr
    if args.s1_data:
        tr_s1 = load_split(sorted(glob.glob(os.path.join(args.s1_data, "ep_*.json"))),
                           args.max_events, aug_gap=args.aug_gap, noid=args.noid)
    print("loaded %d/%d/%d eps (s1 pool %d) (%.1fs)"
          % (len(tr), len(va), len(te), len(tr_s1), time.time() - t0), flush=True)

    model = HomeJepa(d=args.d, layers=args.layers, max_pos=args.max_events + 2,
                     ema=args.ema, room_feats=args.room_feats).to(dev)
    if args.load_s1:
        model.load_state_dict(torch.load(args.load_s1, map_location=dev)["state"])
        print("loaded stage-1 from %s" % args.load_s1, flush=True)
    if args.scratch:
        args.s1_steps = 0
    elif args.load_s1:
        args.s1_steps = args.continue_s1
    print("stage1 params: %.2fM  device: %s" % (
        sum(p.numel() for p in model.parameters()) / 1e6, dev), flush=True)
    pool = [(e, q) for e in range(len(tr)) for q in range(len(tr[e].queries))]
    pool_s1 = [(e, q) for e in range(len(tr_s1)) for q in range(len(tr_s1[e].queries))]

    # ---------------- stage 1: latent pretraining, no labels ----------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    warm = 300
    t0 = time.time()
    acc = []
    for step in range(args.s1_steps):
        for g in opt.param_groups:
            g["lr"] = args.lr * (step + 1) / warm if step < warm else \
                1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(
                    (step - warm) / max(1, args.s1_steps - warm) * 3.141592))
        if args.event_horizons:
            kidx = rng.randrange(3)
            win = ["fut", "fut2", "fut4"][kidx]
        else:
            kidx, win = 0, "fut"
        if args.align_teacher:
            okk = "fut_ok" + win[3:] if win != "fut" else "fut_ok1"
            cand = [p_ for p_ in pool_s1 if tr_s1[p_[0]].queries[p_[1]].get(okk)]
            samp = rng.sample(cand, min(args.bs, len(cand)))
        else:
            samp = rng.sample(pool_s1, args.bs)
        b_ctx = make_batch(tr_s1, samp, dev, window="idxs")
        b_tgt = make_batch(tr_s1, samp, dev, window=win)
        loss, stats = model.stage1_loss(b_ctx, b_tgt, args.w_var, args.w_cov, args.w_anchor,
                                        horizon=kidx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        model.update_target()
        acc.append(stats)
        if (step + 1) % args.val_every == 0 or step == args.s1_steps - 1:
            m = {k: sum(s[k] for s in acc) / len(acc) for k in acc[0]}
            print("s1 %5d  latent %.4f  var %.4f  cov %.4f  anchor %.4f  std %.3f  pr %.1f  (%.0fs)"
                  % (step + 1, m["latent"], m["var"], m["cov"], m["anchor"],
                     m["std"], m["part_ratio"], time.time() - t0), flush=True)
            acc = []
    if args.s1_steps:
        torch.save(dict(state=model.state_dict(), args=vars(args)),
                   os.path.join(args.out, "jepa_%s_s1.pt" % args.tag))
    if args.s2_steps <= 0:      # pretrain-only invocation (stage 2 runs separately)
        print("s1-only run complete", flush=True)
        return

    # ---------------- stage 2: frozen trunk + two-head probe ----------------
    probe = JepaProbe(model, freeze=not args.no_freeze).to(dev)
    tp = [p for p in probe.parameters() if p.requires_grad]
    print("stage2 trainable: %.3fM (%s)" % (sum(p.numel() for p in tp) / 1e6,
          "finetune" if args.no_freeze else "frozen trunk"), flush=True)
    opt = torch.optim.AdamW(tp, lr=args.lr, weight_decay=0.01)
    best, best_state = 1e9, None
    t0 = time.time()
    losses = []
    for step in range(args.s2_steps):
        for g in opt.param_groups:
            g["lr"] = args.lr * (step + 1) / 200 if step < 200 else \
                1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(
                    (step - 200) / max(1, args.s2_steps - 200) * 3.141592))
        b = make_batch(tr, rng.sample(pool, args.bs), dev)
        if args.aux_weight > 0 or args.gate_weight > 0:
            lp, aux, g_logit = probe.all_heads(b)     # single trunk pass
            loss = F.nll_loss(lp, b["gt"])
            if args.aux_weight > 0:
                loss = loss + args.aux_weight * F.nll_loss(aux, b["gt"])
            if args.gate_weight > 0:
                y = (b["gt"] != b["anchor"]).float()
                loss = loss + args.gate_weight * F.binary_cross_entropy_with_logits(g_logit, y)
        else:
            loss = F.nll_loss(probe.log_prob(b), b["gt"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tp, 1.0)
        opt.step()
        losses.append(loss.item())
        if (step + 1) % args.val_every == 0 or step == args.s2_steps - 1:
            vnll, _ = evaluate(probe, va, dev)
            flag = ""
            if vnll < best:
                best, flag = vnll, "  *best"
                best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
                torch.save(dict(state=best_state, args=vars(args), step=step + 1),
                           os.path.join(args.out, "jepa_%s_probe.pt" % args.tag))
            print("s2 %5d  loss %.4f  val_nll %.4f  (%.0fs)%s"
                  % (step + 1, sum(losses) / len(losses), vnll, time.time() - t0, flag), flush=True)
            losses = []

    probe.load_state_dict(best_state)
    tnll, agg = evaluate(probe, te, dev)
    summ = agg.summary()
    print("\ntest NLL %.4f" % tnll)
    print(format_table({args.tag: summ}))
    json.dump(summ, open(os.path.join(args.out, "%s_test.json" % args.tag), "w"), indent=1)
    print("\nsaved -> results/%s_test.json" % args.tag)


if __name__ == "__main__":
    main()

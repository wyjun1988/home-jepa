"""Label-free readout adaptation (P8): freeze the event encoder, finetune the
readout (query/gate/location-side projections) on pseudo-labeled queries built
from the target domain's own logs. Evaluate on the target test set."""
import argparse
import glob
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homejepa.model import load_split, make_batch                  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from reeval import build_supervised, build_jepa_probe              # noqa: E402
from train_supervised import evaluate                              # noqa: E402

FROZEN_PREFIXES = ("enc.", "e_et", "e_cls", "e_cidx", "e_own", "e_rt.", "e_rti",
                   "e_rec", "e_reci", "e_pos", "sca_proj", "norm_in",
                   "trunk.enc", "trunk.e_", "trunk.sca_proj", "trunk.norm_in")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pseudo", default=os.path.join(os.path.dirname(__file__), "..", "data", "homer_v4", "pseudo"))
    ap.add_argument("--test", default=os.path.join(os.path.dirname(__file__), "..", "data", "homer_v4", "test"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dest-weight", type=float, default=1.0,
                    help="weight of the conditional-destination term")
    ap.add_argument("--dest-only", action="store_true",
                    help="adapt ONLY the destination head; leave the gate to the "
                         "label-free recalibration (which is already calibrated)")
    ap.add_argument("--moved-trust", type=float, default=0.64,
                    help="loss weight for pseudo-moved (= measured agreement)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = ap.parse_args()
    dev = torch.device(args.device)
    rng = random.Random(0)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    name = os.path.basename(args.ckpt)
    model, me = (build_jepa_probe if name.startswith("jepa_") else build_supervised)(ck, dev)
    model = model.to(dev)

    frozen = trained = 0
    extra = ("gate.",) if args.dest_only else ()
    for n_, p_ in model.named_parameters():
        if any(n_.startswith(pfx) for pfx in FROZEN_PREFIXES + extra):
            p_.requires_grad_(False)
            frozen += p_.numel()
        else:
            trained += p_.numel()
    print("frozen %.2fM  adapted %.3fM" % (frozen / 1e6, trained / 1e6), flush=True)

    ps = load_split(sorted(glob.glob(os.path.join(args.pseudo, "ep_*.json"))), me, noid=True)
    te = load_split(sorted(glob.glob(os.path.join(args.test, "ep_*.json"))), me, noid=True)
    pool = [(e, q) for e in range(len(ps)) for q in range(len(ps[e].queries))]
    moved_pool = [(e, q) for (e, q) in pool if ps[e].queries[q]["meta"]["moved"]]
    stay_pool = [(e, q) for (e, q) in pool if not ps[e].queries[q]["meta"]["moved"]]
    print("pseudo pool %d (moved %d / stayed %d)  test eps %d"
          % (len(pool), len(moved_pool), len(stay_pool), len(te)), flush=True)

    nll0, agg0 = evaluate(model, te, dev)
    before = agg0.summary()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    model.train()
    for step in range(args.steps):
        # (1) GATE on the natural base rate -- oversampling here would wreck
        # calibration (the failure that collapsed `stayed` in the first run)
        gate_loss = torch.zeros((), device=dev)
        if not args.dest_only:
            bg = make_batch(ps, rng.sample(pool, min(args.bs, len(pool))), dev)
            g_logit, _ = model.parts(bg)
            y = (bg["gt"] != bg["anchor"]).float()
            gate_loss = F.binary_cross_entropy_with_logits(g_logit, y)
        # (2) DESTINATION on oversampled moved pairs -- P(loc | moved) does not
        # depend on the base rate, so oversampling is legitimate here; weight by
        # the measured pseudo-label reliability
        dest_loss = torch.zeros((), device=dev)
        if moved_pool:
            bm = make_batch(ps, rng.sample(moved_pool, min(args.bs, len(moved_pool))), dev)
            _, cond_lp = model.parts(bm)
            dest_loss = args.moved_trust * F.nll_loss(cond_lp, bm["gt"])
        loss = gate_loss + args.dest_weight * dest_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if (step + 1) % 200 == 0:
            print("step %d loss %.4f" % (step + 1, float(loss)), flush=True)

    nll1, agg1 = evaluate(model, te, dev)
    after = agg1.summary()
    # the deployment configuration: adapted destination + label-free gate
    from eval_transfer import eval_recalibrated, log_tau
    taus = [log_tau(ep) for ep in te]
    after_recal = eval_recalibrated(model, te, taus, dev).summary()
    before_recal = None
    out = dict(before=before, after=after, after_recal=after_recal)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
                                     "%s.json" % args.tag), "w"), indent=1)
    for r in ("all", "stayed", "moved", "moved_within", "moved_room"):
        b_, a_, c_ = before.get(r), after.get(r), after_recal.get(r)
        if b_ and a_ and c_:
            print("%-14s t2 %.3f -> %.3f (raw) -> %.3f (+recal)   sr %.2f -> %.2f"
                  % (r, b_["top2"], a_["top2"], c_["top2"], b_["search"], c_["search"]),
                  flush=True)
    print("saved -> results/%s.json" % args.tag)


if __name__ == "__main__":
    main()

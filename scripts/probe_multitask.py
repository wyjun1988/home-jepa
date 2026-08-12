"""World-model scorecard: ONE frozen trunk -> small linear probes for several
downstream tasks (current room, room at +1h, room at +6h, will-it-move-in-6h),
across label fractions. Compares trunks:
  - jepa      : JEPA stage-1 context encoder (pretrained WITHOUT labels)
  - flatsup   : flat_v2 supervised encoder (trained FOR the current-room task)
  - random    : randomly initialized frozen encoder (control)
If the JEPA latent is a real world-model moment, its probes should be
competitive on ALL tasks -- including ones no trunk was trained for -- and
degrade least when labels are scarce."""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homejepa.model import MAX_LOC, load_split, make_batch       # noqa: E402
from homejepa.jepa import HomeJepa                                 # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from train_supervised import MODELS                                # noqa: E402

# event-indexed futures (nmove: destination of the next move, nsight: where
# you will next find it) replace mv6h, which even a random trunk solved
TASKS = ["cur", "1h", "6h", "nmove", "nsight", "tidy", "misplaced"]


def build_trunk(kind, path, dev):
    if kind.startswith("jepa"):
        kind = "jepa"
    if kind == "jepa":
        ck = torch.load(path, map_location=dev)
        a = ck["args"]
        m = HomeJepa(d=a["d"], layers=a["layers"], max_pos=a["max_events"] + 2,
                     room_feats=a.get("room_feats", False))
        # strict=False: pre-horizon checkpoints lack e_horizon (predictor-side,
        # unused here -- we only take the context encoder)
        m.load_state_dict(ck["state"], strict=False)
        trunk = m.context

        def feat(b):
            return trunk.encode(b) + trunk.qdt_proj(b["qdt"])
    elif kind == "flatsup":
        ck = torch.load(path, map_location=dev)
        a = ck["args"]
        trunk = MODELS[a["model"]](d=a["d"], layers=a["layers"],
                                   max_pos=a["max_events"] + 2,
                                   room_feats=a.get("room_feats", False))
        trunk.load_state_dict(ck["state"])

        def feat(b):
            return trunk.encode(b)
    elif kind == "random":
        torch.manual_seed(1234)
        trunk = MODELS["flat"](d=128, layers=4, max_pos=258)

        def feat(b):
            return trunk.encode(b)
    else:
        raise ValueError(kind)
    trunk.to(dev).eval()
    for p in trunk.parameters():
        p.requires_grad_(False)
    return trunk, feat


def extract(feat_fn, eps, dev, batch=256):
    samples = [(e, q) for e in range(len(eps)) for q in range(len(eps[e].queries))]
    feats, labels, masks, metas = [], {t: [] for t in TASKS}, [], []
    with torch.no_grad():
        for i in range(0, len(samples), batch):
            chunk = samples[i:i + batch]
            b = make_batch(eps, chunk, dev)
            feats.append(feat_fn(b).cpu())
            masks.append(b["room_mask"].cpu())
            for ei, qi in chunk:
                q = eps[ei].queries[qi]
                labels["cur"].append(q["gt"])
                for t in TASKS[1:]:
                    labels[t].append(q["futlbl"][t])
                metas.append(q["meta"])
    return (torch.cat(feats), {t: torch.tensor(v) for t, v in labels.items()},
            torch.cat(masks), metas)


def train_probe(X, y, mask, task, dev, steps=1500, bs=1024, lr=1e-2, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = X.shape[1]
    binary = task in ("mv6h", "misplaced")
    W = torch.zeros(d, 1 if binary else MAX_LOC, requires_grad=True)
    bvec = torch.zeros(1 if binary else MAX_LOC, requires_grad=True)
    opt = torch.optim.Adam([W, bvec], lr=lr)
    ok = (y >= 0).nonzero(as_tuple=True)[0]
    for s in range(steps):
        idx = ok[torch.randint(len(ok), (bs,), generator=g)]
        logits = X[idx] @ W + bvec
        if binary:
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), y[idx].float())
        else:
            logits = logits.masked_fill(~mask[idx], -1e9)
            loss = F.cross_entropy(logits, y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach(), bvec.detach()


def eval_probe(W, b, X, y, mask, task, metas):
    ok = (y >= 0)
    logits = X @ W + b
    out = {}
    if task in ("mv6h", "misplaced"):
        p = torch.sigmoid(logits.squeeze(-1))
        pred = (p > 0.5).long()
        out["acc"] = float((pred[ok] == y[ok]).float().mean())
        # balanced accuracy (moved-in-6h is ~30% base rate)
        for c in (0, 1):
            m = ok & (y == c)
            out["acc_c%d" % c] = float((pred[m] == c).float().mean())
        out["bacc"] = 0.5 * (out.pop("acc_c0") + out.pop("acc_c1"))
        return out
    logits = logits.masked_fill(~mask, -1e9)
    lp = torch.log_softmax(logits, -1)
    pred = lp.argmax(-1)
    out["top1"] = float((pred[ok] == y[ok]).float().mean())
    out["nll"] = float(F.nll_loss(lp[ok], y[ok]))
    hm = torch.tensor([1 if (m["moved"] and m.get("mover") not in (None, 0)) else 0
                       for m in metas], dtype=torch.bool) & ok
    if hm.any():
        out["t1_hm"] = float((pred[hm] == y[hm]).float().mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "v1"))
    ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--trunks", default="jepa:results/jepa_jepa_r1_s1.pt,"
                                        "flatsup:results/supervised_flat_r1.pt,random:-")
    ap.add_argument("--fracs", default="1.0,0.1")
    ap.add_argument("--train-eps", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--tag", default="scorecard")
    args = ap.parse_args()
    dev = torch.device(args.device)

    t0 = time.time()
    tr = load_split(sorted(glob.glob(os.path.join(args.data, "train", "ep_*.json")))[:args.train_eps], 256)
    te = load_split(sorted(glob.glob(os.path.join(args.data, "test", "ep_*.json"))), 256)
    print("loaded %d train / %d test eps (%.0fs)" % (len(tr), len(te), time.time() - t0), flush=True)

    rows = {}
    for spec in args.trunks.split(","):
        kind, path = spec.split(":")
        trunk, feat_fn = build_trunk(kind, path, dev)
        t0 = time.time()
        Xtr, ytr, mtr, _ = extract(feat_fn, tr, dev)
        Xte, yte, mte, metas = extract(feat_fn, te, dev)
        print("[%s] features %.0fs  train %s test %s" % (kind, time.time() - t0,
              tuple(Xtr.shape), tuple(Xte.shape)), flush=True)
        del trunk
        for frac in [float(x) for x in args.fracs.split(",")]:
            n = int(len(Xtr) * frac)
            g = torch.Generator().manual_seed(7)
            sub = torch.randperm(len(Xtr), generator=g)[:n]
            for task in TASKS:
                W, b = train_probe(Xtr[sub], ytr[task][sub], mtr[sub], task, dev)
                res = eval_probe(W, b, Xte, yte[task], mte, task, metas)
                rows[(kind, frac, task)] = res
                print("  %-8s frac=%.2f %-5s %s" % (kind, frac, task,
                      " ".join("%s=%.3f" % kv for kv in res.items())), flush=True)

    json.dump({"|".join(map(str, k)): v for k, v in rows.items()},
              open(os.path.join(args.results, "%s.json" % args.tag), "w"), indent=1)
    print("\nsaved -> results/%s.json" % args.tag)


if __name__ == "__main__":
    main()

"""Gate calibration: the model's implied P(moved) = 1 - p(anchor room) vs the
empirical moved rate, reliability-binned + ECE. Product meaning: when the
assistant says "80% it moved", it had better be right 80% of the time."""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homejepa.model import load_split, make_batch                  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from reeval import build_supervised, build_jepa_probe              # noqa: E402


def implied_pmoved(model, eps, dev, batch=256):
    samples = [(e, q) for e in range(len(eps)) for q in range(len(eps[e].queries))]
    conf, lab, dts = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(samples), batch):
            chunk = samples[i:i + batch]
            b = make_batch(eps, chunk, dev)
            p = model.log_prob(b).exp()
            p_anchor = p.gather(1, b["anchor"].view(-1, 1)).squeeze(1).cpu().numpy()
            for j, (ei, qi) in enumerate(chunk):
                q = eps[ei].queries[qi]["meta"]
                conf.append(1.0 - float(p_anchor[j]))
                lab.append(q["moved"])
                dts.append(q["dtbin"])
    return np.array(conf), np.array(lab), np.array(dts)


def report(name, conf, lab):
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    print("\n[%s]  n=%d  base moved rate=%.3f  mean implied P(moved)=%.3f"
          % (name, len(lab), lab.mean(), conf.mean()))
    print("  bin        n      implied  observed")
    for k in range(10):
        m = (conf >= bins[k]) & (conf < bins[k + 1] + (1e-9 if k == 9 else 0))
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(conf[m].mean() - lab[m].mean())
        print("  %.1f-%.1f  %6d   %.3f    %.3f"
              % (bins[k], bins[k + 1], m.sum(), conf[m].mean(), lab[m].mean()))
    print("  ECE = %.4f" % ece)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "v0"))
    ap.add_argument("--models", default="supervised_flat_v2,jepa_ft_probe")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    files = sorted(glob.glob(os.path.join(args.data, "test", "ep_*.json")))
    eps = load_split(files, 256)
    for name in args.models.split(","):
        fp = os.path.join(args.results, name + ".pt")
        ck = torch.load(fp, map_location=dev)
        if name.startswith("jepa_"):
            model, _ = build_jepa_probe(ck, dev)
        else:
            model, _ = build_supervised(ck, dev)
        model = model.to(dev)
        conf, lab, dts = implied_pmoved(model, eps, dev)
        report(name, conf, lab)
        for b in ["1-6h", ">6h"]:
            m = dts == b
            if m.sum():
                report(name + "  dt:" + b, conf[m], lab[m])
        del model


if __name__ == "__main__":
    main()

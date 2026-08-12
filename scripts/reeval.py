"""Re-evaluate saved checkpoints on the test split with the current metric set
(adds top2/top3). Overwrites results/*_test.json."""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homejepa.model import load_split                              # noqa: E402
from homejepa.jepa import HomeJepa, JepaProbe                      # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from train_supervised import MODELS, evaluate                      # noqa: E402


def build_supervised(ck, dev):
    a = ck["args"]
    m = MODELS[a["model"]](d=a["d"], layers=a["layers"],
                           max_pos=a["max_events"] + 2,
                           room_feats=a.get("room_feats", False)).to(dev)
    m.load_state_dict(ck["state"])
    return m, a["max_events"]


def build_jepa_probe(ck, dev):
    a = ck["args"]
    jepa = HomeJepa(d=a["d"], layers=a["layers"], max_pos=a["max_events"] + 2,
                    room_feats=a.get("room_feats", False))
    probe = JepaProbe(jepa, freeze=not a.get("no_freeze", False)).to(dev)
    probe.load_state_dict(ck["state"])
    return probe, a["max_events"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "v0"))
    ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = ap.parse_args()
    dev = torch.device(args.device)
    files = sorted(glob.glob(os.path.join(args.data, "test", "ep_*.json")))
    cache = {}

    def test_eps(me, noid=False):
        if (me, noid) not in cache:
            cache[(me, noid)] = load_split(files, me, noid=noid)
        return cache[(me, noid)]

    for fp in sorted(glob.glob(os.path.join(args.results, "*.pt"))):
        base = os.path.basename(fp)[:-3]
        try:
            ck = torch.load(fp, map_location=dev)
        except Exception as e:
            print("skip %s (%s)" % (base, e))
            continue
        if "state" not in ck or "args" not in ck:
            continue
        try:
            if base.startswith("jepa_") and base.endswith("_probe"):
                model, me = build_jepa_probe(ck, dev)
                tag = base[len("jepa_"):-len("_probe")]
            elif base.startswith("supervised_"):
                model, me = build_supervised(ck, dev)
                tag = base[len("supervised_"):]
            else:
                continue
        except Exception as e:
            print("skip %s (build failed: %s)" % (base, e))
            continue
        nll, agg = evaluate(model, test_eps(me, bool(ck["args"].get("noid", False))), dev)
        summ = agg.summary()
        out = os.path.join(args.results,
                           ("supervised_%s_test.json" if base.startswith("supervised_")
                            else "%s_test.json") % tag)
        json.dump(summ, open(out, "w"), indent=1)
        print("%-28s test NLL %.4f  all-t1 %.3f  -> %s"
              % (tag, nll, summ["all"]["top1"], os.path.basename(out)), flush=True)
        del model


if __name__ == "__main__":
    main()

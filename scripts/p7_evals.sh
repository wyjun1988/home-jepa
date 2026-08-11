#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/p7_evals.log
: > "$LOG"
while ! grep -q "V4_CHAIN_DONE" results/v4_chain.log 2>/dev/null; do sleep 120; done
echo "=== ADT transfer (v4 arms) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/eval_transfer.py --data data/adt/eps/test --tag transfer_adt_v4id \
  --stats baseline_stats_v4.json --models supervised_two_head_v4id 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
$P scripts/eval_transfer.py --data data/adt/eps/test --tag transfer_adt_v4no --noid \
  --stats baseline_stats_v4.json \
  --models supervised_two_head_v4no,jepa_jepa_v4no_ft_probe 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
echo "=== strata reeval (multi-instance) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P - <<'PYEOF' 2>&1 | grep -v "Warning\|warnings" | tee -a "$LOG"
import glob, json, sys, torch
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from homejepa.model import load_split
from reeval import build_supervised, build_jepa_probe
from train_supervised import evaluate
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
for name, noid in [("supervised_two_head_v4id", False), ("supervised_two_head_v4no", True),
                   ("jepa_jepa_v4no_ft_probe", True)]:
    ck = torch.load("results/%s.pt" % name, map_location=dev)
    m, me = (build_jepa_probe if name.startswith("jepa_") else build_supervised)(ck, dev)
    eps = load_split(sorted(glob.glob("data/v4/test/ep_*.json")), me, noid=noid)
    nll, agg = evaluate(m.to(dev), eps, dev)
    s = agg.summary()
    out = name.replace("supervised_", "").replace("jepa_jepa_", "jepa_").replace("_probe", "")
    json.dump(s, open("results/%s_test.json" % out, "w"), indent=1)
    for b in ("cls_single|moved", "cls_multi|moved"):
        if b in s:
            print("%-22s %-18s t1 %.3f t2 %.3f n=%d" % (out, b, s[b]["top1"], s[b]["top2"], s[b]["n"]))
    del m
PYEOF
echo "P7_EVALS_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

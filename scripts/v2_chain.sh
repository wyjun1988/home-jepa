#!/bin/bash
# v1.1 dynamics (lifecycle chains) -> v2 dataset -> retrain -> HOMER transfer.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/v2_chain.log
: > "$LOG"

$P scripts/gen_dataset.py --out data/v2 --split train --homes 1000 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/gen_dataset.py --out data/v2 --split val --homes 40 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/gen_dataset.py --out data/v2 --split test --homes 80 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/audit_dataset.py --data data/v2 2>&1 | grep -v Warning | tail -3 | tee -a "$LOG"
grep -q "PASS: no integrity failures" "$LOG" || { echo AUDIT_FAILED | tee -a "$LOG"; exit 1; }

echo "=== two_head_r2 $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_supervised.py --data data/v2 --tag two_head_r2 --model two_head \
  --steps 3000 --val-every 250 --seed 0 2>&1 | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== jepa_r2 (s1+probe, MPS) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v2 --tag jepa_r2 --device mps \
  --event-horizons --aug-gap 0.3 --s1-steps 3000 --s2-steps 2000 --val-every 250 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== jepa_r2_ft $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v2 --tag jepa_r2_ft --device mps \
  --load-s1 results/jepa_jepa_r2_s1.pt --no-freeze --s2-steps 3000 --val-every 250 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== transfer re-eval $(date +%H:%M:%S) ===" | tee -a "$LOG"
rm -f results/baseline_stats_v2.json
$P - <<'PYEOF' 2>&1 | tee -a "$LOG"
import glob, json, sys
sys.path.insert(0, ".")
from homejepa.baselines import fit_stats
json.dump(fit_stats(sorted(glob.glob("data/v2/train/ep_*.json"))),
          open("results/baseline_stats_v2.json", "w"))
print("v2 baseline stats fitted")
PYEOF
$P scripts/eval_transfer.py --data data/homer/test --tag transfer_homer_v2 \
  --stats baseline_stats_v2.json \
  --models supervised_two_head_r2,jepa_jepa_r2_ft_probe 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "V2_CHAIN_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

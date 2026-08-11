#!/bin/bash
# v1 (receptacle) full pipeline: dataset -> audit gate -> baselines -> MPS
# supervised queue; CPU jepa chain runs in parallel from the same gate.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/v1_chain.log
: > "$LOG"

$P scripts/gen_dataset.py --out data/v1 --split train --homes 1000 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/gen_dataset.py --out data/v1 --split val --homes 40 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/gen_dataset.py --out data/v1 --split test --homes 80 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/audit_dataset.py --data data/v1 2>&1 | grep -v Warning | tail -4 | tee -a "$LOG"
if ! grep -q "PASS: no integrity failures" "$LOG"; then
  echo "AUDIT_FAILED — stopping chain" | tee -a "$LOG"; exit 1
fi
touch results/v1_data_ready

rm -f results/baseline_stats_v1.json
$P - <<'PYEOF' 2>&1 | tee -a "$LOG"
import glob, json, sys
sys.path.insert(0, ".")
from homejepa.baselines import fit_stats
stats = fit_stats(sorted(glob.glob("data/v1/train/ep_*.json")))
json.dump(stats, open("results/baseline_stats_v1.json", "w"))
print("baseline stats fitted")
PYEOF

run() {
  local tag="$1"; shift
  echo "=== $tag  $(date +%H:%M:%S) ===" | tee -a "$LOG"
  $P scripts/train_supervised.py --data data/v1 --tag "$tag" "$@" 2>&1 \
    | grep --line-buffered -v "Warning\|warnings.warn" | tee -a "$LOG"
}
run flat_r1      --model flat     --steps 3000 --val-every 250 --seed 0
run two_head_r1  --model two_head --steps 3000 --val-every 250 --seed 0
run flat_r1_s1   --model flat     --steps 3000 --val-every 250 --seed 1
echo "V1_CHAIN_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

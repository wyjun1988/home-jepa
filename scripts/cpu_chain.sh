#!/bin/bash
# 1x runs with the NEW objective on CPU: isolates the objective effect
# (jepa_v3 vs jepa_v2) independent of data scale.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/cpu_chain.log
: > "$LOG"
export OMP_NUM_THREADS=8

echo "=== jepa_v3 (1x, event horizons + mask) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --tag jepa_v3 --device cpu --threads 8 \
  --event-horizons --aug-gap 0.3 \
  --s1-steps 3000 --s2-steps 2000 --val-every 250 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== jepa_v3_ft $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --tag jepa_v3_ft --load-s1 results/jepa_jepa_v3_s1.pt \
  --no-freeze --device cpu --threads 8 --s2-steps 3000 --val-every 250 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "CPU_CHAIN_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

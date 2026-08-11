#!/bin/bash
# Post-audit rerun, MPS lane. Order = decision value.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/queue5.log
mkdir -p results
: > "$LOG"

run() {
  local tag="$1"; shift
  echo "=== $tag  $(date +%H:%M:%S) ===" | tee -a "$LOG"
  $P scripts/train_supervised.py --tag "$tag" "$@" 2>&1 \
    | grep --line-buffered -v "Warning\|warnings.warn" | tee -a "$LOG"
  echo "=== $tag done $(date +%H:%M:%S) ===" | tee -a "$LOG"
}

run flat_v2      --model flat              --steps 3000 --val-every 250 --seed 0
run flat_v3      --model flat --room-feats --steps 3000 --val-every 250 --seed 0
run attrs_v2     --model two_head --room-feats --max-events 0 --steps 3000 --val-every 500 --seed 0
run two_head_v3  --model two_head --room-feats --steps 3000 --val-every 250 --seed 0
run flat_v2_s1   --model flat              --steps 3000 --val-every 250 --seed 1
run flat_v3_s1   --model flat --room-feats --steps 3000 --val-every 250 --seed 1

echo "QUEUE5_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

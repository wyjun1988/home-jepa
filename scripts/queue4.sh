#!/bin/bash
# flat won the P2-1 comparison, so: flat + negative-evidence attributes (never
# run), then seed repeats -- conclusions have been flipping on single runs.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/queue4.log
mkdir -p results
: > "$LOG"

run() {
  local tag="$1"; shift
  echo "=== $tag  $(date +%H:%M:%S) ===" | tee -a "$LOG"
  $P scripts/train_supervised.py --tag "$tag" "$@" 2>&1 \
    | grep --line-buffered -v "Warning\|warnings.warn" | tee -a "$LOG"
  echo "=== $tag done $(date +%H:%M:%S) ===" | tee -a "$LOG"
}

run flat_v3     --model flat --room-feats --steps 3000 --val-every 250 --seed 0
run flat_v3_s1  --model flat --room-feats --steps 3000 --val-every 250 --seed 1
run flat_v3_s2  --model flat --room-feats --steps 3000 --val-every 250 --seed 2
run flat_v2_s1  --model flat              --steps 3000 --val-every 250 --seed 1

echo "QUEUE4_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

#!/bin/bash
# Trimmed MPS queue: waits for the currently-running two_head_v2 to exit, then
# runs only what is still missing (attrs_only already done on CPU).
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/queue2.log
mkdir -p results
: > "$LOG"

while pgrep -f "train_supervised.py --tag two_head_v2 " > /dev/null; do sleep 20; done
echo "two_head_v2 finished, starting queue2 $(date +%H:%M:%S)" | tee -a "$LOG"

run() {
  local tag="$1"; shift
  echo "=== $tag  $(date +%H:%M:%S) ===" | tee -a "$LOG"
  $P scripts/train_supervised.py --tag "$tag" "$@" 2>&1 \
    | grep --line-buffered -v "Warning\|warnings.warn" | tee -a "$LOG"
  echo "=== $tag done $(date +%H:%M:%S) ===" | tee -a "$LOG"
}

# fair v0 comparison: same 1000-home data, same budget, flat softmax
run flat_v2        --model flat     --steps 3000 --val-every 250 --seed 0
# seed variance on the main config -- single-run gaps have been driving decisions
run two_head_v2_s1 --model two_head --steps 3000 --val-every 250 --seed 1
run two_head_v2_s2 --model two_head --steps 3000 --val-every 250 --seed 2

echo "QUEUE2_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

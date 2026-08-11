#!/bin/bash
# P2-1 experiment queue. Runs sequentially on MPS; each run is independent so a
# failure does not stop the queue.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/overnight.log
mkdir -p results
: > "$LOG"

run() {
  local tag="$1"; shift
  echo "=== $tag  $(date +%H:%M:%S) ===" | tee -a "$LOG"
  $P scripts/train_supervised.py --tag "$tag" "$@" 2>&1 \
    | grep -v "Warning\|warnings.warn" | tee -a "$LOG"
  echo "=== $tag done $(date +%H:%M:%S) ===" | tee -a "$LOG"
}

# 1. main question: does the aux head + 4x data beat v0's 0.290 on housemate-moved?
run two_head_v2      --model two_head --steps 3000 --val-every 250 --seed 0

# 2. fair v0 comparison: same data, same budget, flat softmax
run flat_v2          --model flat     --steps 3000 --val-every 250 --seed 0

# 3. ablation A0: attributes only (anchor + dt), no event history at all.
#    If this matches #1, the 4D event log is not earning its keep.
run attrs_only       --model two_head --steps 3000 --val-every 250 --seed 0 --max-events 0

# 4-5. seed variance on the main config (single-run gaps have been driving
#      decisions; need to know what is noise)
run two_head_v2_s1   --model two_head --steps 3000 --val-every 250 --seed 1
run two_head_v2_s2   --model two_head --steps 3000 --val-every 250 --seed 2

echo "ALL_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

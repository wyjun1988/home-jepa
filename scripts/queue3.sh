#!/bin/bash
# Waits for the running flat_v2, then: room_feats+events (are they complementary?)
# and a seed repeat of the main config.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/queue3.log
mkdir -p results
: > "$LOG"

while pgrep -f "train_supervised.py --tag flat_v2 " > /dev/null; do sleep 20; done
echo "flat_v2 finished, queue3 starts $(date +%H:%M:%S)" | tee -a "$LOG"

run() {
  local tag="$1"; shift
  echo "=== $tag  $(date +%H:%M:%S) ===" | tee -a "$LOG"
  $P scripts/train_supervised.py --tag "$tag" "$@" 2>&1 \
    | grep --line-buffered -v "Warning\|warnings.warn" | tee -a "$LOG"
  echo "=== $tag done $(date +%H:%M:%S) ===" | tee -a "$LOG"
}

# best-model candidate: negative-evidence attributes AND the event sequence
run two_head_v3    --model two_head --room-feats --steps 3000 --val-every 250 --seed 0
# seed variance on the main config
run two_head_v3_s1 --model two_head --room-feats --steps 3000 --val-every 250 --seed 1

echo "QUEUE3_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

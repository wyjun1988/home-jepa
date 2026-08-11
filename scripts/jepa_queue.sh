#!/bin/bash
# Post-audit rerun, CPU lane: JEPA pretrain -> frozen probe, then finetune vs
# scratch control (the actual "does pretraining pay" comparison).
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/jepa_queue.log
: > "$LOG"
export OMP_NUM_THREADS=8

j() {
  local tag="$1"; shift
  echo "=== $tag  $(date +%H:%M:%S) ===" | tee -a "$LOG"
  $P scripts/train_jepa.py --tag "$tag" --device cpu --threads 8 "$@" 2>&1 \
    | grep --line-buffered -v "Warning\|warnings.warn" | tee -a "$LOG"
  echo "=== $tag done $(date +%H:%M:%S) ===" | tee -a "$LOG"
}

j jepa_v2       --s1-steps 3000 --s2-steps 2000 --val-every 250 --seed 0
j jepa_ft       --load-s1 results/jepa_jepa_v2_s1.pt --no-freeze --s2-steps 3000 --val-every 250 --seed 0
j jepa_scratch  --scratch --no-freeze --s2-steps 3000 --val-every 250 --seed 0

echo "JEPA_QUEUE_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

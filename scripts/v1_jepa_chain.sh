#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/v1_jepa_chain.log
: > "$LOG"
export OMP_NUM_THREADS=8
while [ ! -f results/v1_data_ready ]; do sleep 60; done
j() {
  local tag="$1"; shift
  echo "=== $tag  $(date +%H:%M:%S) ===" | tee -a "$LOG"
  $P scripts/train_jepa.py --data data/v1 --tag "$tag" --device cpu --threads 8 "$@" 2>&1 \
    | grep --line-buffered -v "Warning\|warnings.warn" | tee -a "$LOG"
}
j jepa_r1     --event-horizons --aug-gap 0.3 --s1-steps 3000 --s2-steps 2000 --val-every 250 --seed 0
j jepa_r1_ft  --load-s1 results/jepa_jepa_r1_s1.pt --no-freeze --s2-steps 3000 --val-every 250 --seed 0
j jepa_r1_scr --scratch --no-freeze --s2-steps 3000 --val-every 250 --seed 0
echo "V1_JEPA_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

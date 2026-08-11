#!/bin/bash
# Label-free adaptation A/B: continue JEPA stage-1 on HOMER adapt logs (no
# labels), refit the frozen probe on SIM labels, evaluate on HOMER test.
# Control = identical protocol without the continued stage-1.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/adapt_chain.log
: > "$LOG"
export OMP_NUM_THREADS=8

j() {
  local tag="$1"; shift
  echo "=== $tag  $(date +%H:%M:%S) ===" | tee -a "$LOG"
  $P scripts/train_jepa.py --tag "$tag" --device cpu --threads 8 --data data/v1 \
    --test-data data/homer/test "$@" 2>&1 \
    | grep --line-buffered -v "Warning\|warnings.warn" | tee -a "$LOG"
}

j jepa_r1_hctrl --load-s1 results/jepa_jepa_r1_s1.pt --continue-s1 0 \
  --s2-steps 2000 --val-every 250 --seed 0
j jepa_r1_hadapt --load-s1 results/jepa_jepa_r1_s1.pt --continue-s1 1500 \
  --s1-data data/homer/adapt --event-horizons --aug-gap 0.3 \
  --s2-steps 2000 --val-every 250 --seed 0

echo "ADAPT_CHAIN_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

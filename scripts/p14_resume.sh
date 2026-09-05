#!/bin/bash
# P14 재개: 시드2 의 align s1/ft + scr 만. 랩톱 절전 방지를 위해 caffeinate 필수.
cd "$(dirname "$0")/.." || exit 1
P=~/work/home-jepa/.venv/bin/python
LOG=results/p14_seeds.log
S=2
echo "=== [재개] seed $S jepa_align s1 $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v5 --tag jepa_v6_align_s$S --noid \
  --event-horizons --aug-gap 0.3 --align-teacher --s1-steps 3000 --s2-steps 0 --val-every 1000 --seed $S 2>&1 \
  | grep -vE "Warning|warn" | tail -2 | tee -a "$LOG"
echo "=== [재개] seed $S jepa_align_ft $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v5 --tag jepa_v6_align_s${S}_ft --noid \
  --load-s1 results/jepa_jepa_v6_align_s${S}_s1.pt --no-freeze --aux-weight 0.5 --gate-weight 0.5 \
  --s2-steps 5000 --val-every 1000 --seed $S 2>&1 | grep -vE "Warning|warn" | tail -3 | tee -a "$LOG"
echo "=== [재개] seed $S jepa_scr $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v5 --tag jepa_v6_scr_s$S --noid \
  --scratch --no-freeze --aux-weight 0.5 --gate-weight 0.5 --s2-steps 5000 --val-every 1000 --seed $S 2>&1 \
  | grep -vE "Warning|warn" | tail -3 | tee -a "$LOG"
echo "P14_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

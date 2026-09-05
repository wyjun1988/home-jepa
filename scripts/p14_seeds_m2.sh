#!/bin/bash
# P14: P13 판정 시드 확정 (4암 x 시드 1,2). 시드 0 은 로컬 기존 결과 사용.
# 4090 런북 Job1 을 M2(299ms/step)로 대체 수행.
cd "$(dirname "$0")/.." || exit 1
P=~/work/home-jepa/.venv/bin/python
LOG=results/p14_seeds.log
: > "$LOG"
for S in 1 2; do
  echo "=== [seed $S] supervised $(date +%H:%M:%S) ===" | tee -a "$LOG"
  $P scripts/train_supervised.py --data data/v5 --tag two_head_v5_s$S --model two_head \
    --noid --gate-weight 0.5 --steps 5000 --val-every 1000 --seed $S 2>&1 | grep -vE "Warning|warn" | tail -3 | tee -a "$LOG"
  for ARM in base align; do
    EXTRA=""; [ "$ARM" = "align" ] && EXTRA="--align-teacher"
    echo "=== [seed $S] jepa_$ARM s1 $(date +%H:%M:%S) ===" | tee -a "$LOG"
    $P scripts/train_jepa.py --data data/v5 --tag jepa_v6_${ARM}_s$S --noid \
      --event-horizons --aug-gap 0.3 $EXTRA --s1-steps 3000 --s2-steps 0 --val-every 1000 --seed $S 2>&1 \
      | grep -vE "Warning|warn" | tail -2 | tee -a "$LOG"
    echo "=== [seed $S] jepa_${ARM}_ft $(date +%H:%M:%S) ===" | tee -a "$LOG"
    $P scripts/train_jepa.py --data data/v5 --tag jepa_v6_${ARM}_s${S}_ft --noid \
      --load-s1 results/jepa_jepa_v6_${ARM}_s${S}_s1.pt --no-freeze --aux-weight 0.5 --gate-weight 0.5 \
      --s2-steps 5000 --val-every 1000 --seed $S 2>&1 | grep -vE "Warning|warn" | tail -3 | tee -a "$LOG"
  done
  echo "=== [seed $S] jepa_scr $(date +%H:%M:%S) ===" | tee -a "$LOG"
  $P scripts/train_jepa.py --data data/v5 --tag jepa_v6_scr_s$S --noid \
    --scratch --no-freeze --aux-weight 0.5 --gate-weight 0.5 --s2-steps 5000 --val-every 1000 --seed $S 2>&1 \
    | grep -vE "Warning|warn" | tail -3 | tee -a "$LOG"
done
echo "P14_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

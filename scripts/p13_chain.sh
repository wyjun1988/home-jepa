#!/bin/bash
# P13: teacher-alignment A/B. Both arms share the aug-glance leak fix; the only
# difference is --align-teacher (s1 trains only on (query,k) whose k-th reveal
# matches the current state). Tests CONCEPT_REVIEW 2.1's falsifiable claim.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/p13_chain.log
: > "$LOG"

for ARM in base align; do
  EXTRA=""; [ "$ARM" = "align" ] && EXTRA="--align-teacher"
  echo "=== jepa_v6_${ARM} s1 $(date +%H:%M:%S) ===" | tee -a "$LOG"
  $P scripts/train_jepa.py --data data/v5 --tag jepa_v6_${ARM} --device mps --noid \
    --event-horizons --aug-gap 0.3 $EXTRA --s1-steps 3000 --s2-steps 0 --val-every 500 --seed 0 2>&1 \
    | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
  echo "=== jepa_v6_${ARM}_ft $(date +%H:%M:%S) ===" | tee -a "$LOG"
  $P scripts/train_jepa.py --data data/v5 --tag jepa_v6_${ARM}_ft --device mps --noid \
    --load-s1 results/jepa_jepa_v6_${ARM}_s1.pt --no-freeze --aux-weight 0.5 --gate-weight 0.5 \
    --s2-steps 5000 --val-every 500 --seed 0 2>&1 \
    | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
done

echo "=== ref-past (신규 암 + 기준) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/eval_refpast.py --tag refpast_v6 \
  --ckpts supervised_two_head_v5,jepa_jepa_v6_base_ft_probe,jepa_jepa_v6_align_ft_probe 2>&1 \
  | grep -v "Warning\|warnings" | tee -a "$LOG"
echo "P13_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

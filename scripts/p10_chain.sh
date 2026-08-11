#!/bin/bash
# P10 (GPU): v5 data (L1 person presence) + gate supervision; then absence-
# insistence audit, HOMER transfer, and the room-only pseudo adaptation A/B.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/p10_chain.log
: > "$LOG"

echo "=== two_head_v5 (gate-sup + people) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_supervised.py --data data/v5 --tag two_head_v5 --model two_head --noid \
  --gate-weight 0.5 --steps 5000 --val-every 500 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== two_head_v5_nogate (ablation: 게이트감독 없음) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_supervised.py --data data/v5 --tag two_head_v5_nogate --model two_head --noid \
  --steps 5000 --val-every 500 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== jepa_v5 s1 $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v5 --tag jepa_v5 --device mps --noid \
  --event-horizons --aug-gap 0.3 --s1-steps 3000 --s2-steps 0 --val-every 500 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
echo "=== jepa_v5_ft $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v5 --tag jepa_v5_ft --device mps --noid \
  --load-s1 results/jepa_jepa_v5_s1.pt --no-freeze --s2-steps 5000 --val-every 500 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== 부재-고집 계측 $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/diag_answerable.py --data data/v5/test \
  --ckpts supervised_two_head_v5,supervised_two_head_v5_nogate,jepa_jepa_v5_ft_probe 2>&1 \
  | grep -v "Warning\|warnings" | tee -a "$LOG"

echo "=== HOMER 전이 + 적응 A/B $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/eval_transfer.py --data data/homer_v5/test --tag transfer_homer_v5 --noid \
  --stats baseline_stats_v5.json --models supervised_two_head_v5,jepa_jepa_v5_ft_probe 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
$P scripts/adapt_readout.py --ckpt results/supervised_two_head_v5.pt \
  --pseudo data/homer_v5/pseudo --test data/homer_v5/test \
  --tag adapt_v5_roomonly --dest-only --steps 600 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "P10_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

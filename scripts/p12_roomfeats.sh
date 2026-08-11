#!/bin/bash
# P12: turn the per-location combination attributes back ON (--room-feats) and
# redo the 3-way fair comparison. Decides (a) whether the glance/absence
# attributes earn their place, (b) whether JEPA pretraining survives with the
# full input.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/p12_chain.log
: > "$LOG"

echo "=== two_head_v5_rf (supervised, room-feats ON) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_supervised.py --data data/v5 --tag two_head_v5_rf --model two_head --noid \
  --room-feats --gate-weight 0.5 --steps 5000 --val-every 500 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== jepa_v5_rf s1 $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v5 --tag jepa_v5_rf --device mps --noid --room-feats \
  --event-horizons --aug-gap 0.3 --s1-steps 3000 --s2-steps 0 --val-every 500 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
echo "=== jepa_v5_rf_ft $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v5 --tag jepa_v5_rf_ft --device mps --noid --room-feats \
  --load-s1 results/jepa_jepa_v5_rf_s1.pt --no-freeze --aux-weight 0.5 --gate-weight 0.5 \
  --s2-steps 5000 --val-every 500 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
echo "=== jepa_v5_rf_scr (랜덤초기화 대조) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v5 --tag jepa_v5_rf_scr --device mps --noid --room-feats \
  --scratch --no-freeze --aux-weight 0.5 --gate-weight 0.5 \
  --s2-steps 5000 --val-every 500 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== 계측 $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/diag_answerable.py --data data/v5/test \
  --ckpts supervised_two_head_v5_rf,jepa_jepa_v5_rf_ft_probe,jepa_jepa_v5_rf_scr_probe 2>&1 \
  | grep -v "Warning\|warnings" | tee -a "$LOG"

echo "=== HOMER 전이 + 적응 $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/eval_transfer.py --data data/homer_v5/test --tag transfer_homer_v5_rf --noid \
  --stats baseline_stats_v5.json \
  --models supervised_two_head_v5_rf,jepa_jepa_v5_rf_ft_probe 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
$P scripts/adapt_readout.py --ckpt results/supervised_two_head_v5_rf.pt \
  --pseudo data/homer_v5/pseudo --test data/homer_v5/test \
  --tag adapt_v5_rf --dest-only --steps 600 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "P12_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

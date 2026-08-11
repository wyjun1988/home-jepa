#!/bin/bash
# P9: anchor fix (class-level SELF_DROP) + anchor-instance absence features
# wired into the gate. Retrain noid arms, then re-run the answerability audit.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/p9_chain.log
: > "$LOG"

echo "=== two_head_v4no3 $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_supervised.py --data data/v4 --tag two_head_v4no3 --model two_head --noid \
  --steps 5000 --val-every 500 --seed 0 2>&1 | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== jepa_v4no3 s1 $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v4 --tag jepa_v4no3 --device mps --noid \
  --event-horizons --aug-gap 0.3 --s1-steps 3000 --s2-steps 0 --val-every 500 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
echo "=== jepa_v4no3_ft $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v4 --tag jepa_v4no3_ft --device mps --noid \
  --load-s1 results/jepa_jepa_v4no3_s1.pt --no-freeze --s2-steps 5000 --val-every 500 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== answerability audit (부재확인시 원위치 고집 여부) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/diag_answerable.py --ckpts supervised_two_head_v4no3,jepa_jepa_v4no3_ft_probe 2>&1 \
  | grep -v "Warning\|warnings" | tee -a "$LOG"

echo "=== HOMER transfer + adaptation $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/eval_transfer.py --data data/homer_v4/test --tag transfer_homer_v4no3 --noid \
  --stats baseline_stats_v4.json --models supervised_two_head_v4no3 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
$P scripts/adapt_readout.py --ckpt results/supervised_two_head_v4no3.pt \
  --tag adapt_two_head_v4no3 --dest-only --steps 600 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "P9_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

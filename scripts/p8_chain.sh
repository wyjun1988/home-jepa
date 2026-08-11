#!/bin/bash
# P8: noid slack recovery (hand-chain anchor + 5000 steps) -> pseudo-label
# readout adaptation on target logs -> HOMER test A/B.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/p8_chain.log
: > "$LOG"

$P scripts/convert_homer.py --split train --user-seeds 1 --out data/homer_v4 2>&1 | tail -1 | tee -a "$LOG"

echo "=== two_head_v4no2 (hand-chain, 5000) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_supervised.py --data data/v4 --tag two_head_v4no2 --model two_head --noid \
  --steps 5000 --val-every 500 --seed 0 2>&1 | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== jepa_v4no2 s1 $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v4 --tag jepa_v4no2 --device mps --noid \
  --event-horizons --aug-gap 0.3 --s1-steps 3000 --s2-steps 0 --val-every 500 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
echo "=== jepa_v4no2_ft (5000) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v4 --tag jepa_v4no2_ft --device mps --noid \
  --load-s1 results/jepa_jepa_v4no2_s1.pt --no-freeze --s2-steps 5000 --val-every 500 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== HOMER transfer (pre-adapt) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/eval_transfer.py --data data/homer_v4/test --tag transfer_homer_v4no2 --noid \
  --stats baseline_stats_v4.json \
  --models supervised_two_head_v4no2,jepa_jepa_v4no2_ft_probe 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== pseudo queries $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/pseudo_queries.py 2>&1 | tee -a "$LOG"

echo "=== readout adaptation A/B $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/adapt_readout.py --ckpt results/supervised_two_head_v4no2.pt \
  --tag adapt_two_head_v4no2 2>&1 | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
$P scripts/adapt_readout.py --ckpt results/jepa_jepa_v4no2_ft_probe.pt \
  --tag adapt_jepa_v4no2_ft 2>&1 | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "P8_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

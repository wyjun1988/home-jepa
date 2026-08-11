#!/bin/bash
# P11: fair JEPA-vs-supervised A/B. Stage 2 now gets the same two auxiliary
# signals Track S has had (aux flat-softmax 0.5, gate BCE 0.5). Same data,
# same steps, same lr; only the trunk's pretraining differs.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/p11_fair.log
: > "$LOG"

echo "=== jepa_v5_ft_fair (aux+gate parity) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v5 --tag jepa_v5_ft_fair --device mps --noid \
  --load-s1 results/jepa_jepa_v5_s1.pt --no-freeze \
  --aux-weight 0.5 --gate-weight 0.5 --s2-steps 5000 --val-every 500 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== jepa_v5_scr_fair (사전학습 없음 대조) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v5 --tag jepa_v5_scr_fair --device mps --noid \
  --scratch --no-freeze --aux-weight 0.5 --gate-weight 0.5 \
  --s2-steps 5000 --val-every 500 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== 3자 계측 $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/diag_answerable.py --data data/v5/test \
  --ckpts supervised_two_head_v5,jepa_jepa_v5_ft_fair_probe,jepa_jepa_v5_scr_fair_probe 2>&1 \
  | grep -v "Warning\|warnings" | tee -a "$LOG"

echo "=== HOMER 전이 + 적응 $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/eval_transfer.py --data data/homer_v5/test --tag transfer_homer_v5_fair --noid \
  --stats baseline_stats_v5.json --models jepa_jepa_v5_ft_fair_probe 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
$P scripts/adapt_readout.py --ckpt results/jepa_jepa_v5_ft_fair_probe.pt \
  --pseudo data/homer_v5/pseudo --test data/homer_v5/test \
  --tag adapt_v5_jepa_fair --dest-only --steps 600 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "P11_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

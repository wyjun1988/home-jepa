#!/bin/bash
# JEPA scaling chain v2 -- NEW objective (event-indexed horizons k in {1,2,4}
# + structured gap masking), 5x unlabeled data, then finetune on the same
# labeled 1000 homes, then the event-task scorecard.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/scaling_chain.log
: > "$LOG"

while ! grep -q "5000/5000" results/gen_pretrain.log 2>/dev/null; do sleep 30; done
[ -e data/v0_pretrain/train ] || ln -s pretrain data/v0_pretrain/train
echo "pool ready $(date +%H:%M:%S)" | tee -a "$LOG"
$P scripts/audit_dataset.py --data data/v0_pretrain --train-sample 60 --regen-check 2 2>&1 \
  | grep -v "Warning\|warnings" | tail -4 | tee -a "$LOG"

echo "=== s1_5x (event horizons + structured mask) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --tag jepa_5x --data data/v0_pretrain --device mps \
  --event-horizons --aug-gap 0.3 \
  --s1-steps 3000 --s2-steps 0 --val-every 250 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== ft_5x $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --tag jepa_5x_ft --load-s1 results/jepa_jepa_5x_s1.pt \
  --no-freeze --device mps --s2-steps 3000 --val-every 250 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== scorecard v2 (event tasks, all trunks) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/probe_multitask.py --tag scorecard_v2 \
  --trunks jepa5x:results/jepa_jepa_5x_s1.pt,jepa1x:results/jepa_jepa_v3_s1.pt,jepa_old:results/jepa_jepa_v2_s1.pt,flatsup:results/supervised_flat_v2.pt,random:- 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "SCALING_CHAIN_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

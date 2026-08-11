#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/recover.log
: > "$LOG"
while ! grep -q "CPU_CHAIN_DONE" results/cpu_chain.log 2>/dev/null; do sleep 60; done
echo "=== reeval new fts $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/reeval.py 2>&1 | grep -v "Warning\|warnings" | tee -a "$LOG"
echo "=== scorecard_v2 $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/probe_multitask.py --tag scorecard_v2 \
  --trunks jepa5x:results/jepa_jepa_5x_s1.pt,jepa1x:results/jepa_jepa_v3_s1.pt,jepa_old:results/jepa_jepa_v2_s1.pt,flatsup:results/supervised_flat_v2.pt,random:- 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
echo "RECOVER_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

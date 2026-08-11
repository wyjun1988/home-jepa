#!/bin/bash
# LLM-pack pilot: pack-generated sim -> retrain -> HOMER transfer A/B vs r2.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/v3_chain.log
: > "$LOG"

$P scripts/gen_dataset.py --out data/v3 --split train --homes 600 --packs data/packs/pilot_packs.json 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/gen_dataset.py --out data/v3 --split val --homes 36 --packs data/packs/pilot_packs.json 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/gen_dataset.py --out data/v3 --split test --homes 48 --packs data/packs/pilot_packs.json 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/audit_dataset.py --data data/v3 2>&1 | grep -v Warning | tail -3 | tee -a "$LOG"
grep -q "PASS: no integrity failures" "$LOG" || { echo AUDIT_FAILED | tee -a "$LOG"; exit 1; }

echo "=== two_head_r3 $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_supervised.py --data data/v3 --tag two_head_r3 --model two_head \
  --steps 3000 --val-every 250 --seed 0 2>&1 | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

echo "=== jepa_r3 (s1, MPS) $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v3 --tag jepa_r3 --device mps \
  --event-horizons --aug-gap 0.3 --s1-steps 3000 --s2-steps 0 --val-every 250 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
echo "=== jepa_r3_ft $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v3 --tag jepa_r3_ft --device mps \
  --load-s1 results/jepa_jepa_r3_s1.pt --no-freeze --s2-steps 3000 --val-every 250 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

rm -f results/baseline_stats_v3.json
$P - <<'PYEOF' 2>&1 | tee -a "$LOG"
import glob, json, sys
sys.path.insert(0, ".")
from homejepa.baselines import fit_stats
json.dump(fit_stats(sorted(glob.glob("data/v3/train/ep_*.json"))),
          open("results/baseline_stats_v3.json", "w"))
print("v3 stats fitted")
PYEOF
$P scripts/eval_transfer.py --data data/homer/test --tag transfer_homer_v3 \
  --stats baseline_stats_v3.json \
  --models supervised_two_head_r3,jepa_jepa_r3_ft_probe 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
echo "V3_CHAIN_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

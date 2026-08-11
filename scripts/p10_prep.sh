#!/bin/bash
# CPU-only prep for P10: regenerate v5 (with L1 person presence), audit,
# rebuild pseudo-labels (room-moves only) and verify their quality.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/p10_prep.log
: > "$LOG"
rm -rf data/v5 data/homer_v5
$P scripts/gen_dataset.py --out data/v5 --split train --homes 600 --packs data/packs/pilot_packs.json 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/gen_dataset.py --out data/v5 --split val   --homes 36 --packs data/packs/pilot_packs.json 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/gen_dataset.py --out data/v5 --split test  --homes 48 --packs data/packs/pilot_packs.json 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/audit_dataset.py --data data/v5 2>&1 | grep -v Warning | tail -3 | tee -a "$LOG"
$P scripts/convert_homer.py --out data/homer_v5 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/convert_homer.py --split train --user-seeds 1 --out data/homer_v5 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/audit_dataset.py --data data/homer_v5 --train-sample 40 --regen-check 0 2>&1 | grep -v Warning | tail -2 | tee -a "$LOG"
rm -f results/baseline_stats_v5.json
$P - <<'PYEOF' 2>&1 | tee -a "$LOG"
import glob, json, sys
sys.path.insert(0, ".")
from homejepa.baselines import fit_stats
json.dump(fit_stats(sorted(glob.glob("data/v5/train/ep_*.json"))), open("results/baseline_stats_v5.json","w"))
print("v5 stats fitted")
PYEOF
echo "--- pseudo labels: room-moves only ---" | tee -a "$LOG"
$P scripts/pseudo_queries.py --src data/homer_v5/adapt --out data/homer_v5/pseudo 2>&1 | grep -v Warning | tee -a "$LOG"
$P scripts/verify_pseudo.py --src data/homer_v5/adapt --pseudo data/homer_v5/pseudo 2>&1 | grep -v Warning | tee -a "$LOG"
echo "P10_PREP_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

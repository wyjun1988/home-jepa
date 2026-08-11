#!/bin/bash
# v4: packs + glance attributes + blended lifecycle chains; id vs noid A/B.
cd "$(dirname "$0")/.." || exit 1
P=/Users/wooyeol/work/stock-v2/.venv-mps/bin/python
LOG=results/v4_chain.log
: > "$LOG"

$P scripts/gen_dataset.py --out data/v4 --split train --homes 600 --packs data/packs/pilot_packs.json 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/gen_dataset.py --out data/v4 --split val --homes 36 --packs data/packs/pilot_packs.json 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/gen_dataset.py --out data/v4 --split test --homes 48 --packs data/packs/pilot_packs.json 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/audit_dataset.py --data data/v4 2>&1 | grep -v Warning | tail -3 | tee -a "$LOG"
grep -q "PASS: no integrity failures" "$LOG" || { echo AUDIT_FAILED | tee -a "$LOG"; exit 1; }
$P scripts/convert_homer.py --out data/homer_v4 2>&1 | tail -1 | tee -a "$LOG"
$P scripts/audit_dataset.py --data data/homer_v4 --train-sample 0 --regen-check 0 2>&1 | grep -v Warning | tail -2 | tee -a "$LOG"

run() { local tag="$1"; shift
  echo "=== $tag  $(date +%H:%M:%S) ===" | tee -a "$LOG"
  $P scripts/train_supervised.py --data data/v4 --tag "$tag" "$@" 2>&1 \
    | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"; }

run two_head_v4id --model two_head --steps 3000 --val-every 250 --seed 0
run two_head_v4no --model two_head --noid --steps 3000 --val-every 250 --seed 0

echo "=== jepa_v4no s1 $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v4 --tag jepa_v4no --device mps --noid \
  --event-horizons --aug-gap 0.3 --s1-steps 3000 --s2-steps 0 --val-every 250 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
echo "=== jepa_v4no_ft $(date +%H:%M:%S) ===" | tee -a "$LOG"
$P scripts/train_jepa.py --data data/v4 --tag jepa_v4no_ft --device mps --noid \
  --load-s1 results/jepa_jepa_v4no_s1.pt --no-freeze --s2-steps 3000 --val-every 250 --seed 0 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"

rm -f results/baseline_stats_v4.json
$P - <<'PYEOF' 2>&1 | tee -a "$LOG"
import glob, json, sys
sys.path.insert(0, ".")
from homejepa.baselines import fit_stats
json.dump(fit_stats(sorted(glob.glob("data/v4/train/ep_*.json"))),
          open("results/baseline_stats_v4.json", "w"))
print("v4 stats fitted")
PYEOF
$P scripts/eval_transfer.py --data data/homer_v4/test --tag transfer_homer_v4id \
  --stats baseline_stats_v4.json --models supervised_two_head_v4id 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
$P scripts/eval_transfer.py --data data/homer_v4/test --tag transfer_homer_v4no --noid \
  --stats baseline_stats_v4.json \
  --models supervised_two_head_v4no,jepa_jepa_v4no_ft_probe 2>&1 \
  | grep --line-buffered -v "Warning\|warnings" | tee -a "$LOG"
echo "V4_CHAIN_DONE $(date +%H:%M:%S)" | tee -a "$LOG"

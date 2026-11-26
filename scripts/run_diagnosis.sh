#!/usr/bin/env bash
# Why the gain is small: diagnosis on the development rooms (run scripts/run_indoor.sh first).
set -euo pipefail
cd "$(dirname "$0")/.."
D=${FVS_DATA:-data}; O=${FVS_OUT:-outputs}
python -m fvs.evaluate.diagnosis --tier sparse --feat $O/indoor/dev_sparse/offline/feat.pkl \
  --rec $O/indoor/dev_sparse/offline/rec.pkl --models $D/planners --out $O/results/diagnosis_sparse.json
python -m fvs.evaluate.diagnosis --tier dense --feat $O/indoor/dev_dense/offline/feat.pkl \
  --rec $O/indoor/dev_dense/offline/rec.pkl --models $D/planners --out $O/results/diagnosis_dense.json

#!/usr/bin/env bash
# Object experiment (sparse and dense objects). Requires ShapeNetCore.v2 under data/external or SHAPENET_ROOT.
set -euo pipefail
cd "$(dirname "$0")/.."
D=${FVS_DATA:-data}; O=${FVS_OUT:-outputs}
mkdir -p $O/results
# sparse objects: 130 test objects in 6 shards
for i in 0 1 2 3 4 5; do
  python -m fvs.replay.objects_sparse --shard $i,6 --out $O/objects_sparse
done
python -m fvs.evaluate.objects $O/objects_sparse --kind sparse --out $O/results/objects_sparse.json
# dense objects: 66 objects
python -m fvs.replay.objects_dense --out $O/objects_dense
python -m fvs.evaluate.objects $O/objects_dense --kind dense --out $O/results/objects_dense.json

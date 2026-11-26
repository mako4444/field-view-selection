#!/usr/bin/env bash
# Indoor experiment on SceneFun3D: sparse and dense tiers, test and development rooms.
# Requires the SceneFun3D videos listed in data/indoor/{test,dev}_videos.txt under data/external/scenefun3d
# or SCENEFUN3D_ROOT. The shard counts are part of the protocol (decision seed = 0 + position within the shard).
set -euo pipefail
cd "$(dirname "$0")/.."
D=${FVS_DATA:-data}; O=${FVS_OUT:-outputs}; M=$D/planners
mkdir -p $O/results

sparse() {  # sparse <split> <shards>
  local s=$1 n=$2 R=$O/indoor/$1_sparse
  for i in $(seq 0 $(($n - 1))); do
    python -m fvs.replay.indoor_sparse --crops $D/indoor/$s --list $D/indoor/$s.txt --shard $i/$n --out $R/replay
  done
  python -m fvs.offline.sparse $R/replay --crops $D/indoor/$s --lwl --out $R/offline
  python -m fvs.offline.angle $R/replay --crops $D/indoor/$s --out $R/angle.pkl
  python -m fvs.planners.hgb apply --tier sparse --models $M --feat $R/offline/feat.pkl --out $R/planners.pkl
  python -m fvs.planners.lwl predict --models $M/lwl_sparse_all_points --data $R/offline/lwl --out $R/lwl.pkl
  local E="--tier sparse --rec $R/offline/rec.pkl --plm $R/planners.pkl --angle $R/angle.pkl"
  python -m fvs.evaluate.indoor $E --lwl $R/lwl.pkl --out $O/results/${s}_sparse_field.json
  python -m fvs.evaluate.indoor $E --method harris --out $O/results/${s}_sparse_harris.json
  for h in 50 100; do
    python -m fvs.evaluate.indoor $E --mm $h --out $O/results/${s}_sparse_field_h$h.json
  done
}

dense() {  # dense <split> <shards>
  local s=$1 n=$2 R=$O/indoor/$1_dense
  python -m fvs.prepare_dense --crops $D/indoor/$s --list $D/indoor/$s.txt
  for i in $(seq 0 $(($n - 1))); do
    python -m fvs.replay.indoor_dense --crops $D/indoor/$s --list $D/indoor/$s.txt --shard $i/$n --out $R/replay
  done
  python -m fvs.offline.dense $R/replay --crops $D/indoor/$s --out $R/offline
  python -m fvs.offline.angle $R/replay --crops $D/indoor/$s --methods harris --out $R/angle.pkl
  python -m fvs.planners.hgb apply --tier dense --models $M --feat $R/offline/feat.pkl --out $R/planners.pkl
  python -m fvs.planners.lwl predict --models $M/lwl_dense_all_points --data $R/offline --grid grid_all.npy --out $R/lwl_all.pkl
  python -m fvs.planners.lwl predict --models $M/lwl_dense_localizer --data $R/offline --grid grid.npy --out $R/lwl_localizer.pkl
  local E="--tier dense --rec $R/offline/rec.pkl --plm $R/planners.pkl --angle $R/angle.pkl --covis $R/replay/rows_s*.csv"
  python -m fvs.evaluate.indoor $E --lwl $R/lwl_all.pkl --out $O/results/${s}_dense.json
  python -m fvs.evaluate.indoor $E --lwl $R/lwl_localizer.pkl --out $O/results/${s}_dense_lwl_localizer.json
}

sparse test 4
sparse dev 2
dense test 6
dense dev 4

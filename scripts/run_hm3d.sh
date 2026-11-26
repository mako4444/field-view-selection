#!/usr/bin/env bash
# Transfer to HM3D (scenes 00017 and 00270 of the ActLoc release; HM3D_ACTLOC_ROOT or data/external/hm3d_actloc).
set -euo pipefail
cd "$(dirname "$0")/.."
D=${FVS_DATA:-data}; O=${FVS_OUT:-outputs}; M=$D/planners; R=$O/hm3d
mkdir -p $O/results
for s in 00017-oEPjPNSPmzL 00270-bDTsgcSK5Qr; do
  python -m fvs.hm3d.render_capture $s --capture $R/capture
done
python -m fvs.hm3d.make_crops 00017-oEPjPNSPmzL 00270-bDTsgcSK5Qr --capture $R/capture --out $R/crops
for i in 0 1; do
  python -m fvs.replay.indoor_sparse --crops $R/crops --list $D/hm3d/crops.txt --frames $R/capture --shard $i/2 \
    --out $R/replay
done
python -m fvs.offline.sparse $R/replay --crops $R/crops --frames $R/capture --out $R/offline
python -m fvs.planners.hgb apply --tier sparse --models $M --feat $R/offline/feat.pkl --out $R/planners.pkl
python -m fvs.evaluate.indoor --tier sparse --rec $R/offline/rec.pkl --plm $R/planners.pkl --level crop \
  --out $O/results/hm3d_sparse_field.json

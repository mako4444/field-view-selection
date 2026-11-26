#!/usr/bin/env bash
# Figures of two test decisions of the indoor experiment (run scripts/run_indoor.sh first).
set -euo pipefail
cd "$(dirname "$0")/.."
O=${FVS_OUT:-outputs}
python scripts/fig_indoor.py --pick 466112/c002:11 --view 1.0,0.5,3.5 --fdepth 0.45 --out $O/figures/teaser.pdf
python scripts/fig_indoor.py --pick 438101/c003:11 --view 1.0,2.5,1.0 --fdepth 0.45 --out $O/figures/qual.pdf

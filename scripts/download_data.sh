#!/usr/bin/env bash
# Downloads the data of this repository from Zenodo (doi:10.5281/zenodo.24573550),
# checks the archives and unpacks them into data/. Run from the repository root.
set -euo pipefail
URL=https://zenodo.org/records/24573550/files
PARTS="weights planners objects indoor hm3d"
DEST=${1:-data}

mkdir -p "$DEST"
cd "$DEST"
curl -fL -o SHA256SUMS "$URL/SHA256SUMS?download=1"
for p in $PARTS; do
  if [ -d "$p" ]; then echo "$p/ exists, skipped"; continue; fi
  curl -fL -o "$p.tar.gz" "$URL/$p.tar.gz?download=1"
  grep "  $p.tar.gz\$" SHA256SUMS | sha256sum -c -
  tar xzf "$p.tar.gz"
  rm "$p.tar.gz"
done

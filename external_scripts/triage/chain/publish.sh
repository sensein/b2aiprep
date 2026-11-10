#!/bin/bash
# Copy a finished, checked triage run into a derivatives folder:   bash publish.sh SETTINGS DST [--replace]
# Never part of the chain: run it by hand, after the check stage passed and the tables were looked at.
#
#   1. the run roots, per subject (symlinks followed, so DST holds files only);
#   2. the tables, their dictionary, the viewer and dataset_description.json (finalize_derivatives.py);
#   3. a count of files and bytes in both trees (senselab's verify_derivatives_copy), which must match.
#
# DST must be empty or absent, unless --replace, which deletes from DST what the run does not hold. Built release
# trees are read only: publish into a scratch copy and move it deliberately.
source ~/.bashrc 2>/dev/null || true
export SETTINGS=$(readlink -f "${1:?usage: publish.sh SETTINGS DST [--replace]}")
DST=${2:?usage: publish.sh SETTINGS DST [--replace]}
REPLACE=${3:-}
CHAIN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$CHAIN_DIR/common.sh"
set -euo pipefail

if [ -d "$DST" ] && [ -n "$(ls -A "$DST")" ] && [ "$REPLACE" != --replace ]; then
  echo "FATAL: $DST is not empty; pass --replace to overwrite it" >&2
  exit 2
fi
mkdir -p "$DST"
log "copying run roots $RUNS -> $DST"
for sub in "$RUNS"/sub-*; do
  rsync -rltL --delete-during "$sub/" "$DST/$(basename "$sub")/"
done
"$PY" "$CHAIN_DIR/finalize_derivatives.py" --vectors "$VECTORS" --dst "$DST" --bids-root "$BIDS_ROOT" \
  --commit "$COMMIT" --profile "$PROFILE"
drive verify_derivatives_copy --src "$RUNS" --dst "$DST" --threads "${SLURM_CPUS_PER_TASK:-16}" \
  --needle "$WORK" --json "$WORK/logs/publish_verify.json"
log "published to $DST"

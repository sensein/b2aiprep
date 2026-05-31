#!/usr/bin/env bash
# Generate a Synapse sync manifest as a batch job. This walks the BIDS folder and
# creates/resolves the folder structure on Synapse, then writes the manifest TSV.
#
# Usage:
#   sbatch --export=ALL sage_generate.sbatch <bids_folder> <manifest_file> [extra args, e.g. --adult]
#
#SBATCH --job-name=sage_generate
#SBATCH --output=/orcd/data/satra/002/datasets/b2aivoice/b2aiprep/logs/sage_generate_%j.out
#SBATCH --error=/orcd/data/satra/002/datasets/b2aivoice/b2aiprep/logs/sage_generate_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=pi_satra
#SBATCH --mail-user=wilke18@mit.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

BIDS="${1:?Usage: sbatch sage_generate.sbatch <bids_folder> <manifest_file> [extra args]}"
MANIFEST="${2:?Usage: sbatch sage_generate.sbatch <bids_folder> <manifest_file> [extra args]}"
shift 2   # remaining args (e.g. --adult) pass through to the generate script

PY=/home/wilke18/.conda/envs/b2aiprep_test/bin/python
SCRIPT_DIR=/orcd/data/satra/002/datasets/b2aivoice/b2aiprep/external_scripts/sage_upload_scripts
cd "$SCRIPT_DIR"   # so load_dotenv() picks up SAGE_PAT

$PY sage_generate_manifest.py --bids_folder "$BIDS" --manifest_file "$MANIFEST" "$@"
echo "=== manifest line count ==="
wc -l "$MANIFEST"

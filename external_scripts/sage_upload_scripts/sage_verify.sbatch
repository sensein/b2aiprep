#!/usr/bin/env bash
# Full local-vs-Synapse verification of a dataset (md5-level), as a batch job.
#
# verify_sage_contents.py reads local files directly and fetches each file's md5
# from the Synapse server, so it is independent of the local ~/.synapseCache.
# This is the authoritative "did everything upload correctly" check.
#
# Usage:
#   sbatch --export=ALL sage_verify.sbatch <bids_folder> [extra verify args, e.g. --adult]
#
#SBATCH --job-name=sage_verify
#SBATCH --output=/orcd/data/satra/002/datasets/b2aivoice/b2aiprep/logs/sage_verify_%j.out
#SBATCH --error=/orcd/data/satra/002/datasets/b2aivoice/b2aiprep/logs/sage_verify_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=pi_satra
#SBATCH --mail-user=wilke18@mit.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

BIDS="${1:?Usage: sbatch sage_verify.sbatch <bids_folder> [extra verify args]}"
shift   # remaining args (e.g. --adult) pass through to the verify script

PY=/home/wilke18/.conda/envs/b2aiprep_test/bin/python
SCRIPT_DIR=/orcd/data/satra/002/datasets/b2aivoice/b2aiprep/external_scripts/sage_upload_scripts
cd "$SCRIPT_DIR"   # so load_dotenv() picks up SAGE_PAT

# Read-only check (no --sync): reports md5 mismatches, missing, and extra files.
$PY verify_sage_contents.py --bids_folder "$BIDS" --get_md5 "$@"

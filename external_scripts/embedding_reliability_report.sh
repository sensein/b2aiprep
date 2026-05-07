#!/bin/bash
# Run the embedding reliability report (synthetic mixture evaluation).
#
# Requires speaker profiles to be built first:
#   sbatch --array=1-20 build_speaker_profiles_array.sh
#
# Submit with a dependency on the profile build array job:
#   sbatch --dependency=afterok:<PROFILE_ARRAY_JOB_ID> embedding_reliability_report.sh
# or run manually after profiles are complete:
#   sbatch embedding_reliability_report.sh
#
#SBATCH -c 4
#SBATCH -t 4:00:00
#SBATCH -p pi_satra,ou_bcs_normal,mit_normal
#SBATCH --mem-per-cpu=10G
#SBATCH -o logs/embedding_reliability_%j.out
#SBATCH -e logs/embedding_reliability_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wilke18@mit.edu

BIDS_DIR="/orcd/data/satra/002/datasets/b2aivoice/post_3.0/v3.1/peds/bids_04_03_26"
PROFILES_DIR="/orcd/data/satra/002/datasets/b2aivoice/speaker_profiles/peds"
OUTPUT_DIR="/orcd/data/satra/002/datasets/b2aivoice/speaker_profiles/peds/reliability_report"
PIPELINE_CONFIG=""  # leave empty to use built-in defaults

# Optional: path to a BIDS directory of adult recordings to use as intruder pool.
# When set, all intruders are drawn from this pool and tagged "adult" in the report,
# enabling evaluation of the peds+adult mixture scenario in addition to peds-only.
# Leave empty to use other peds participants as intruders (tagged "peds").
INTRUDER_BIDS_DIR=""

# Intruder placement positions to evaluate (comma-separated: start, middle, end).
# More positions → more mixtures and longer runtime.
INTRUDER_POSITIONS="end"

# Threshold selection: maximum acceptable false-negative rate.
# Lower = stricter (smaller review queue); see research.md Decision 9.
MAX_FNR_TARGET=0.05

source "$HOME/.bashrc"
conda activate b2aiprep_test

set -euo pipefail

mkdir -p "${OUTPUT_DIR}" logs

CONFIG_FLAG=""
if [ -n "${PIPELINE_CONFIG}" ]; then
    CONFIG_FLAG="--pipeline-config ${PIPELINE_CONFIG}"
fi

INTRUDER_BIDS_FLAG=""
if [ -n "${INTRUDER_BIDS_DIR}" ]; then
    INTRUDER_BIDS_FLAG="--intruder-bids-dir ${INTRUDER_BIDS_DIR}"
fi

echo "[$(date)] Starting embedding reliability report"

b2aiprep-cli embedding-reliability-report \
    "${BIDS_DIR}" \
    "${PROFILES_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-format both \
    --max-fnr-target "${MAX_FNR_TARGET}" \
    --intruder-ratios "0.10,0.20,0.40" \
    --intruder-snr-db "0,5,10" \
    --intruder-positions "${INTRUDER_POSITIONS}" \
    ${INTRUDER_BIDS_FLAG} \
    ${CONFIG_FLAG} \
    --log-level INFO

echo "[$(date)] Done. Report written to: ${OUTPUT_DIR}"

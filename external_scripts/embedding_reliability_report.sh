#!/bin/bash
# Run the embedding reliability report (synthetic mixture evaluation).
#
# THREE-SCENARIO APPROACH (see spec.md FR-011):
#
#   Scenario 1 — adult→peds (PRIMARY, drives threshold selection):
#     Target profiles = peds; intruder pool = adult BIDS dir.
#     Set INTRUDER_BIDS_DIR to the adult dataset path.
#     The "adult" row in the Intruder-Type Breakdown is labelled "adult→peds (primary)".
#
#   Scenario 2 — peds→peds (same-cohort baseline):
#     Leave INTRUDER_BIDS_DIR empty; intruders are drawn from other peds participants.
#     Run with peds BIDS_DIR and peds PROFILES_DIR.
#
#   Scenario 3 — adult→adult (sanity-check baseline):
#     Set both BIDS_DIR and PROFILES_DIR to adult paths; leave INTRUDER_BIDS_DIR empty.
#
# To run all three scenarios, submit three separate jobs with the appropriate config.
# The recommended thresholds in the final report should be taken from Scenario 1.
#
# Requires speaker profiles to be built first:
#   sbatch --array=1-20 build_speaker_profiles_array.sh        # peds
#   sbatch --array=1-20 build_speaker_profiles_array_adult.sh  # adult
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

# Optional: path to adult BIDS dir for adult→peds (Scenario 1, primary).
# Leave empty for same-cohort intruders (Scenario 2: peds→peds).
INTRUDER_BIDS_DIR=""

# Intruder placement positions to evaluate (comma-separated: start, middle, end).
INTRUDER_POSITIONS="end"

# Threshold selection: maximum acceptable false-negative rate (see research.md Decision 9).
MAX_FNR_TARGET=0.05

# Optional: peds release exclusion list for real-data validation section (peds-only).
# Path to a JSON file containing a list of file stems to treat as uncertain positives.
# Leave empty to produce a synthetic-only report (valid and complete without it).
EXCLUSION_LIST=""

# Optional: Evans model predictions CSV (columns: file_path, y_pred, confidence, uncertainty).
# Only used when EXCLUSION_LIST is set. Leave empty to omit Evans recall from the report.
EVANS_PREDICTIONS=""

# Optional: train-split annotation CSV to exclude train-contaminated Evans predictions.
# Only used when EVANS_PREDICTIONS is set.
EVANS_TRAIN_ANNOTATIONS=""

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

EXCLUSION_LIST_FLAG=""
if [ -n "${EXCLUSION_LIST}" ]; then
    EXCLUSION_LIST_FLAG="--exclusion-list ${EXCLUSION_LIST}"
fi

EVANS_PREDICTIONS_FLAG=""
if [ -n "${EVANS_PREDICTIONS}" ]; then
    EVANS_PREDICTIONS_FLAG="--evans-predictions ${EVANS_PREDICTIONS}"
fi

EVANS_TRAIN_ANNOTATIONS_FLAG=""
if [ -n "${EVANS_TRAIN_ANNOTATIONS}" ]; then
    EVANS_TRAIN_ANNOTATIONS_FLAG="--evans-train-annotations ${EVANS_TRAIN_ANNOTATIONS}"
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
    ${EXCLUSION_LIST_FLAG} \
    ${EVANS_PREDICTIONS_FLAG} \
    ${EVANS_TRAIN_ANNOTATIONS_FLAG} \
    ${CONFIG_FLAG} \
    --log-level INFO

echo "[$(date)] Done. Report written to: ${OUTPUT_DIR}"

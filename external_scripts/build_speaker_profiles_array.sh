#!/bin/bash
# SLURM array job: build per-participant speaker profiles across N shards.
#
# Usage:
#   mkdir -p logs
#   sbatch --array=1-20 build_speaker_profiles_array.sh
#
# After all tasks complete, run embedding_reliability_report.sh.
# NUM_PARTS below must match the --array upper bound above.
#
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH -p pi_satra,ou_bcs_normal,mit_normal
#SBATCH --mem-per-cpu=10G
#SBATCH -o logs/build_profiles_%A_%a.out
#SBATCH -e logs/build_profiles_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wilke18@mit.edu

BIDS_DIR="/orcd/data/satra/002/datasets/b2aivoice/post_3.0/v3.1/peds/bids_04_03_26"
PROFILES_DIR="/orcd/data/satra/002/datasets/b2aivoice/speaker_profiles/peds"
NUM_PARTS=20        # must match --array=1-N above
PIPELINE_CONFIG=""  # leave empty to use built-in defaults

source "$HOME/.bashrc"
conda activate b2aiprep_test

set -euo pipefail

mkdir -p "${PROFILES_DIR}" logs

CONFIG_FLAG=""
if [ -n "${PIPELINE_CONFIG}" ]; then
    CONFIG_FLAG="--pipeline-config ${PIPELINE_CONFIG}"
fi

echo "[$(date)] Starting profile build shard ${SLURM_ARRAY_TASK_ID} of ${NUM_PARTS}"

b2aiprep-cli build-speaker-profiles \
    "${BIDS_DIR}" \
    "${PROFILES_DIR}" \
    --part "${SLURM_ARRAY_TASK_ID}" \
    --num-parts "${NUM_PARTS}" \
    ${CONFIG_FLAG} \
    --log-level INFO

echo "[$(date)] Shard ${SLURM_ARRAY_TASK_ID} complete"

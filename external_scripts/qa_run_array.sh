#!/bin/bash
# SLURM array job: run the b2aiprep audio QA pipeline across N shards.
#
# Usage:
#   mkdir -p logs
#   sbatch --array=1-20 qa_run_array.sh
#
# After all array tasks complete, run qa_run_merge.sh to combine shard outputs.
# NUM_PARTS below must match the --array upper bound above.
#
#SBATCH -c 4
#SBATCH -t 12:00:00
#SBATCH -p pi_satra,ou_bcs_normal,mit_normal
#SBATCH --mem=250G
#SBATCH -o logs/qa_run_%A_%a.out
#SBATCH -e logs/qa_run_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wilke18@mit.edu

# Configure these before submitting
BIDS_DIR="/orcd/data/satra/002/datasets/b2aivoice/post_3.0/v3.1/peds/bids_04_03_26"
OUTPUT_DIR="/orcd/data/satra/002/datasets/b2aivoice/qa_run_results/peds"
NUM_PARTS=20                        # must match --array=1-N above
PIPELINE_CONFIG=""                  # leave empty to use built-in defaults

source "$HOME/.bashrc"
conda activate b2aiprep_test

set -euo pipefail

mkdir -p "${OUTPUT_DIR}/shard_${SLURM_ARRAY_TASK_ID}"

CONFIG_FLAG=""
if [ -n "${PIPELINE_CONFIG}" ]; then
    CONFIG_FLAG="--pipeline-config ${PIPELINE_CONFIG}"
fi

echo "[$(date)] Starting shard ${SLURM_ARRAY_TASK_ID} of ${NUM_PARTS}"

b2aiprep-cli qa-run \
    "${BIDS_DIR}" \
    "${OUTPUT_DIR}/shard_${SLURM_ARRAY_TASK_ID}" \
    --part "${SLURM_ARRAY_TASK_ID}" \
    --num-parts "${NUM_PARTS}" \
    ${CONFIG_FLAG} \
    --log-level INFO

echo "[$(date)] Shard ${SLURM_ARRAY_TASK_ID} complete"

#!/bin/bash
# Merge per-shard QA outputs into a single set of TSVs in OUTPUT_DIR.
#
# Run this after all qa_run_array.sh array tasks complete:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> qa_run_merge.sh
# or manually:
#   bash qa_run_merge.sh
#
# Expects shard subdirectories at OUTPUT_DIR/shard_1/, shard_2/, etc.

OUTPUT_DIR="/orcd/data/satra/002/datasets/b2aivoice/qa_run_results/peds"

#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH -p mit_normal
#SBATCH --mem-per-cpu=8G
#SBATCH -o logs/qa_merge_%j.out
#SBATCH -e logs/qa_merge_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wilke18@mit.edu

source "$HOME/.bashrc"
conda activate b2aiprep_test

set -euo pipefail

echo "[$(date)] Merging shard outputs in ${OUTPUT_DIR}"

# Files to merge (header from shard_1, data rows from all shards)
for TSV in qa_check_results.tsv qa_composite_scores.tsv needs_review_queue.tsv; do
    MERGED="${OUTPUT_DIR}/${TSV}"
    FIRST=1
    for SHARD_DIR in "${OUTPUT_DIR}"/shard_*/; do
        SHARD_FILE="${SHARD_DIR}${TSV}"
        if [ ! -f "${SHARD_FILE}" ]; then
            continue
        fi
        if [ "${FIRST}" -eq 1 ]; then
            cp "${SHARD_FILE}" "${MERGED}"
            FIRST=0
        else
            # Append without header (skip first line)
            tail -n +2 "${SHARD_FILE}" >> "${MERGED}"
        fi
    done
    if [ "${FIRST}" -eq 1 ]; then
        echo "WARNING: no shard produced ${TSV}" >&2
    else
        ROWS=$(wc -l < "${MERGED}")
        echo "  ${TSV}: $((ROWS - 1)) data rows"
    fi
done

# Copy one config snapshot to the top-level output dir (all shards use same config)
FIRST_CONFIG=$(find "${OUTPUT_DIR}/shard_1" -name "qa_pipeline_config_*.json" 2>/dev/null | head -1)
if [ -n "${FIRST_CONFIG}" ]; then
    cp "${FIRST_CONFIG}" "${OUTPUT_DIR}/"
    echo "  config: $(basename ${FIRST_CONFIG})"
fi

echo "[$(date)] Merge complete — outputs in ${OUTPUT_DIR}"

#!/bin/bash
#SBATCH -c 4
#SBATCH -t 4:00:00
#SBATCH -p pi_satra,ou_bcs_normal,mit_normal
#SBATCH --mem-per-cpu=10G
#SBATCH -o logs/compare_speaker_detection_%j.out
#SBATCH -e logs/compare_speaker_detection_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wilke18@mit.edu

BIDS_DIR="/orcd/data/satra/002/datasets/b2aivoice/post_3.0/v3.1/peds/bids_04_03_26"
EXCLUSION_JSON="/orcd/data/satra/002/datasets/b2aivoice/post_3.0/config/shared_release_config_peds_all/audio_filestems_to_remove.json"
EVANS_PREDICTIONS="/orcd/data/satra/002/datasets/b2aivoice/b2ai-model/b2ai-models/eval_outputs_adult_uncertainty_all/predictions_with_uncertainty.csv"
EVANS_TRAIN_ANNOTATIONS="/orcd/data/satra/002/datasets/b2aivoice/b2ai-model/b2ai-models/annotations/train/peds_annotations_20000.csv"
OUTPUT_TSV="/orcd/data/satra/002/datasets/b2aivoice/compare_speaker_detection_results.tsv"

source "$HOME/.bashrc"
conda activate b2aiprep_test

set -euo pipefail

mkdir -p logs

echo "[$(date)] Starting speaker detection comparison"

python /orcd/data/satra/002/datasets/b2aivoice/b2aiprep/external_scripts/compare_speaker_detection.py \
    "${BIDS_DIR}" \
    "${EXCLUSION_JSON}" \
    --evans-predictions "${EVANS_PREDICTIONS}" \
    --evans-train-annotations "${EVANS_TRAIN_ANNOTATIONS}" \
    --output "${OUTPUT_TSV}" \
    --no-progress

echo "[$(date)] Done. Results at: ${OUTPUT_TSV}"

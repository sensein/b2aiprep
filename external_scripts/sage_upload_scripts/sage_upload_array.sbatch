#!/usr/bin/env bash
# Per-subject parallel upload of a BIDS dataset to Sage/Synapse.
#
# Each array task uploads exactly one subject's subtree, so tasks never write to
# the same Synapse folder concurrently (the only shared parent is the dataset
# folder, which already exists). This avoids the folder-creation races that the
# old --start/--end row-splitting approach hit.
#
# Usage:
#   1) Generate the manifest once (login node is fine):
#        python sage_generate_manifest.py --bids_folder <BIDS> --manifest_file manifest.tsv [--adult]
#   2) Build the subject list:
#        ls -d <BIDS>/sub-* | xargs -n1 basename > subjects.txt
#   3) Submit, sized to the number of subjects, capped at 20 concurrent (%20):
#        sbatch --array=0-$(($(wc -l < subjects.txt) - 1))%20 \
#               sage_upload_array.sbatch manifest.tsv subjects.txt
#   4) Upload the dataset-level (non-subject) files once, after the array:
#        python sage_upload_manifest.py --manifest_file manifest.tsv --toplevel
#   5) Verify:
#        python verify_sage_contents.py --bids_folder <BIDS> [--adult] --get_md5
#
#SBATCH --job-name=sage_upload
#SBATCH --output=/orcd/data/satra/002/datasets/b2aivoice/b2aiprep/logs/sage_upload_%A_%a.out
#SBATCH --error=/orcd/data/satra/002/datasets/b2aivoice/b2aiprep/logs/sage_upload_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
# One subject per task is small, so 4h is generous. Partition options:
#   pi_satra        (up to 48h)
#   ou_bcs_normal   (up to 24h) -- raise --time below if a task needs longer
#SBATCH --partition=pi_satra

set -euo pipefail

MANIFEST="${1:?Usage: sbatch sage_upload_array.sbatch <manifest.tsv> <subjects.txt>}"
SUBJECTS="${2:?Usage: sbatch sage_upload_array.sbatch <manifest.tsv> <subjects.txt>}"

PY=/home/wilke18/.conda/envs/b2aiprep_test/bin/python
SCRIPT_DIR=/orcd/data/satra/002/datasets/b2aivoice/b2aiprep/external_scripts/sage_upload_scripts
cd "$SCRIPT_DIR"   # so load_dotenv() picks up the local .env with SAGE_PAT

# Pick this task's subject (subjects.txt is one subject per line, 0-based index).
SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SUBJECTS")
if [[ -z "$SUBJECT" ]]; then
    echo "No subject for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

echo "Array task $SLURM_ARRAY_TASK_ID -> uploading $SUBJECT"
$PY sage_upload_manifest.py --manifest_file "$MANIFEST" --subject "$SUBJECT"

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
#   3) Submit. The QOS caps submitted jobs per user (pi_satra: 256), so put
#      CHUNK subjects per task to keep the array element count under that.
#      ntasks = ceil(nsubjects / CHUNK); %20 caps how many run at once.
#        N=$(wc -l < subjects.txt); CHUNK=2
#        sbatch --array=0-$(( (N + CHUNK - 1)/CHUNK - 1 ))%20 \
#               sage_upload_array.sbatch manifest.tsv subjects.txt $CHUNK
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
# Email on completion/failure of the array as a whole (no ARRAY_TASKS => not one
# email per subject).
#SBATCH --mail-user=wilke18@mit.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

MANIFEST="${1:?Usage: sbatch sage_upload_array.sbatch <manifest.tsv> <subjects.txt> [chunk]}"
SUBJECTS="${2:?Usage: sbatch sage_upload_array.sbatch <manifest.tsv> <subjects.txt> [chunk]}"
CHUNK="${3:-1}"   # subjects handled per array task (raise it to stay under the QOS submit cap)

PY=/home/wilke18/.conda/envs/b2aiprep_test/bin/python
SCRIPT_DIR=/orcd/data/satra/002/datasets/b2aivoice/b2aiprep/external_scripts/sage_upload_scripts
cd "$SCRIPT_DIR"   # so load_dotenv() picks up the local .env with SAGE_PAT

# This task handles subjects [START..END] (1-based lines of subjects.txt).
START=$(( SLURM_ARRAY_TASK_ID * CHUNK + 1 ))
END=$(( START + CHUNK - 1 ))
SUBJECT_LIST=$(sed -n "${START},${END}p" "$SUBJECTS")
if [[ -z "$SUBJECT_LIST" ]]; then
    echo "No subjects for array index $SLURM_ARRAY_TASK_ID (lines ${START}-${END})"
    exit 0
fi

# Upload each subject in the chunk sequentially (disjoint subjects across tasks,
# so no two tasks ever write to the same Synapse folder). Don't let one subject's
# failure (e.g. a 412 update conflict) abort the rest of the chunk; flag the task
# as failed at the end instead, and re-run stragglers found by verify.
rc=0
for SUBJECT in $SUBJECT_LIST; do
    echo "Array task $SLURM_ARRAY_TASK_ID -> uploading $SUBJECT"
    if ! $PY sage_upload_manifest.py --manifest_file "$MANIFEST" --subject "$SUBJECT"; then
        echo "WARNING: subject $SUBJECT failed; continuing with the rest of the chunk" >&2
        rc=1
    fi
done
exit $rc

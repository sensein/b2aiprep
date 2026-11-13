#!/bin/bash
# Submit one stage of the triage chain and, unless --only, a link job that submits the next stage once every task of
# this one succeeded:   bash submit.sh /abs/path/settings.sh STAGE [--only]
#
# Only one stage is queued at a time, so the chain stays under the per-user submit cap whatever the slice counts.
# A failed task leaves the link unsatisfied (it is cancelled): read the stage's logs under $WORK/slurm/, fix the cause,
# and submit the same stage again; every pass resumes. `bash submit.sh SETTINGS inputs` starts a run from the top.
# Stages, in order: see STAGES below and README.md.
set -euo pipefail
SETTINGS=$(readlink -f "${1:?usage: submit.sh SETTINGS STAGE [--only]}")
STAGE=${2:?usage: submit.sh SETTINGS STAGE [--only]}
ONLY=${3:-}
export SETTINGS
CHAIN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
set +u
source "$CHAIN_DIR/common.sh" >/dev/null
set -u

# The chain goes nowords -> review_opinion -> cohort_collect; review and opinion are the same two steps as separate
# stages (review -> opinion -> cohort_collect), for submitting by hand.
# stage | next stage | slices setting (empty: one job) | rows file the stage reads (empty: none) | CPU/GPU | cpus | mem | time
STAGES="
inputs|fresh|||cpu|1|4G|0:30:00
fresh|runs|FRESH_SLICES|$INPUT|cpu|8|40G|12:00:00
runs|session|||cpu|1|8G|1:00:00
session|replay|SESSION_SLICES|$MANIFEST|cpu|2|12G|4:00:00
replay|review_rows|REPLAY_SLICES|$MANIFEST|cpu|4|24G|12:00:00
review_rows|nowords|||cpu|16|32G|2:00:00
nowords|review_opinion|NOWORDS_SLICES|$REVIEW_ROWS/rows_nowords.jsonl|cpu|2|8G|4:00:00
review_opinion|cohort_collect|REVIEW_OPINION_SLICES|$REVIEW_ROWS/rows_words.jsonl|gpu|8|96G|12:00:00
review|opinion|REVIEW_SLICES|$REVIEW_ROWS/rows_words.jsonl|gpu|8|96G|8:00:00
opinion|cohort_collect|OPINION_SLICES|$REVIEW_ROWS/rows_words.jsonl|gpu|8|96G|8:00:00
cohort_collect|cohort_quantiles|COHORT_COLLECT_SLICES|$MANIFEST|cpu|1|8G|2:00:00
cohort_quantiles|cohort_apply|||cpu|2|16G|0:30:00
cohort_apply|cohort_consistency|COHORT_APPLY_SLICES|$MANIFEST|cpu|4|24G|12:00:00
cohort_consistency|refold|||cpu|1|4G|0:30:00
refold|task_audio|REFOLD_SLICES|$MANIFEST|cpu|2|16G|12:00:00
task_audio|task_squim|TASK_AUDIO_SLICES|$MANIFEST|cpu|2|8G|6:00:00
task_squim|vectors|SQUIM_SLICES|$MANIFEST|cpu|2|8G|12:00:00
vectors|vectors_merge|VECTORS_SLICES|$MANIFEST|cpu|2|24G|12:00:00
vectors_merge|table|||cpu|2|16G|0:30:00
table|check|||cpu|16|64G|4:00:00
check||||cpu|1|8G|0:30:00
"

row=$(printf '%s\n' "$STAGES" | awk -F'|' -v s="$STAGE" '$1 == s')
[ -n "$row" ] || { echo "unknown stage $STAGE; stages: $(printf '%s\n' "$STAGES" | cut -d'|' -f1 | xargs)" >&2; exit 2; }
IFS='|' read -r _ NEXT SLICES_VAR ROWS KIND CPUS MEM TIME <<<"$row"

common=(--job-name="triage-$STAGE" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" --export=ALL,SETTINGS="$SETTINGS")
[ -n "${MAIL:-}" ] && common+=(--mail-type=FAIL --mail-user="$MAIL")
[ -n "${EXCLUDE_NODES:-}" ] && common+=(--exclude="$EXCLUDE_NODES")
if [ "$KIND" = gpu ]; then
  common+=(--partition="$GPU_PARTITION" --gres="$GPU_GRES")
else
  common+=(--partition="$CPU_PARTITION")
fi

submit_next() {  # $1: the job the link waits for (empty: none)
  [ -n "$NEXT" ] && [ "$ONLY" != --only ] || return 0
  dep=()
  [ -n "${1:-}" ] && dep=(--dependency=afterok:"$1" --kill-on-invalid-dep=yes)
  [ -n "${EXCLUDE_NODES:-}" ] && dep+=(--exclude="$EXCLUDE_NODES")
  sbatch --parsable "${dep[@]}" --job-name="triage-link-$NEXT" --partition="$LINK_PARTITION" --cpus-per-task=1 \
    --mem=1G --time=0:10:00 --output="$WORK/slurm/%x_%j.out" --export=ALL,SETTINGS="$SETTINGS" \
    --wrap="bash $CHAIN_DIR/submit.sh $SETTINGS $NEXT"
}

slices=1
if [ -n "$SLICES_VAR" ]; then
  slices=${!SLICES_VAR:?$SLICES_VAR is not set in $SETTINGS}
  if [ -n "$ROWS" ]; then
    [ -f "$ROWS" ] || { echo "FATAL: $STAGE reads $ROWS, which does not exist; run the stage before it" >&2; exit 2; }
    rows=$(grep -c . "$ROWS" || true)
    if [ "$rows" -eq 0 ]; then
      log "$STAGE: $ROWS is empty; nothing to do, going on to ${NEXT:-the end}" | tee -a "$WORK/jobs.txt"
      link=$(submit_next "")
      [ -n "${link:-}" ] && log "  link to $NEXT: $link" | tee -a "$WORK/jobs.txt"
      exit 0
    fi
    [ "$rows" -lt "$slices" ] && slices=$rows
  fi
  job=$(sbatch --parsable "${common[@]}" --array="0-$((slices - 1))%$THROTTLE" --requeue \
    --output="$WORK/slurm/%x_%A_%a.out" --wrap="bash $CHAIN_DIR/step.sh $STAGE \$SLURM_ARRAY_TASK_ID $slices")
else
  job=$(sbatch --parsable "${common[@]}" --output="$WORK/slurm/%x_%j.out" --wrap="bash $CHAIN_DIR/step.sh $STAGE")
fi
link=$(submit_next "$job")
log "$STAGE: job $job ($slices task(s))${link:+, link to $NEXT: $link}" | tee -a "$WORK/jobs.txt"

#!/bin/bash
# One stage of the triage chain, or one slice of it: bash step.sh STAGE [SLICE_INDEX SLICE_COUNT]
# submit.sh runs it as a Slurm job (an array task for a sliced stage, which takes its index from
# SLURM_ARRAY_TASK_ID); it can equally be run by hand. Every pass resumes: a slice run again redoes only what an
# earlier attempt did not finish. The stages and their order are in README.md.
source ~/.bashrc 2>/dev/null || true
: "${SETTINGS:?SETTINGS must name the settings file of the run}"
STAGE=${1:?usage: step.sh STAGE [SLICE_INDEX SLICE_COUNT]}
I=${2:-${SLURM_ARRAY_TASK_ID:-0}}
N=${3:-1}
CHAIN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$CHAIN_DIR/common.sh"
if [ -n "${FFMPEG_MODULE:-}" ]; then module add "$FFMPEG_MODULE" 2>/dev/null || true; fi
set -uo pipefail

export SENSELAB_RUN_ID="triage-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
GPU_STAGES=" review opinion "
[[ "$GPU_STAGES" == *" $STAGE "* ]] || export CUDA_VISIBLE_DEVICES=""
slice=(--slice-index "$I" --slice-count "$N")
profile=(--profile "$PROFILE")
log "stage $STAGE slice $I/$N host $(hostname) commit $COMMIT profile $PROFILE"

case "$STAGE" in
inputs)
  "$PY" "$CHAIN_DIR/make_input_manifest.py" "$BIDS_ROOT" --out "$INPUT" ${STEMS_FILE:+--stems "$STEMS_FILE"}
  ;;
fresh)   # the whole graph per recording, reviewer off, into $FRESH (resumable). A recording that fails is its row;
         # the runs stage is the gate, so a task exits 0 unless its arguments could not be resolved (2).
  drive triage_corpus run "$INPUT" --out-root "$FRESH" "${slice[@]}" --hints "$HINTS" "${profile[@]}" --commit "$COMMIT"
  rc=$?
  [ "$rc" -eq 1 ] && { log "fresh: some recordings failed; their rows say why (the runs stage reports them)"; rc=0; }
  exit "$rc"
  ;;
runs)    # the finished runs as the passes' manifest; fails where a recording has no finished run, unless ALLOW_MISSING=1
  drive triage_corpus manifest "$FRESH" --out "$MANIFEST" --expect "$INPUT"
  rc=$?
  [ "$rc" -eq 1 ] && [ "${ALLOW_MISSING:-0}" = 1 ] && { log "runs: recordings with no finished run left out (ALLOW_MISSING=1)"; rc=0; }
  exit "$rc"
  ;;
session) # SESSION's floor, one BIDS session at a time
  drive extend_session_floor "$MANIFEST" "${slice[@]}" --log-dir "$LOGS/session" "${profile[@]}"
  ;;
replay)  # re-decide from TAXONOMY now the session floors stand; exit 3 (reviewer readings to redo) is a result
  drive extend_replay_decisions "$MANIFEST" "${slice[@]}" --hints "$HINTS" --log-dir "$LOGS/replay" \
    --commit "$COMMIT" "${profile[@]}"
  rc=$?
  [ "$rc" -eq 3 ] && { log "replay: some readings need the reviewer again (exit 3)"; rc=0; }
  exit "$rc"
  ;;
review_rows)  # every row the replay did not read (a fresh corpus was never reviewed), split into words (GPU) and nowords (CPU)
  drive triage_review_rows rereads "$MANIFEST" --replay-logs "$LOGS/replay" --out "$REVIEW_ROWS/rows_needs_reread.jsonl" \
    --all-unread &&
    drive triage_review_rows split "$REVIEW_ROWS/rows_needs_reread.jsonl" --out-dir "$REVIEW_ROWS" "${profile[@]}"
  ;;
nowords) # REVIEW and SECOND_OPINION recorded without a model for rows with no words (CPU)
  drive extend_llm_review "$REVIEW_ROWS/rows_nowords.jsonl" "${slice[@]}" --config "$RUN_CONFIG" --hints "$HINTS" \
    --log-dir "$LOGS/nowords" --no-refold "${profile[@]}"
  rc_review=$?
  drive extend_second_opinion "$REVIEW_ROWS/rows_nowords.jsonl" "${slice[@]}" --config "$RUN_CONFIG" --hints "$HINTS" \
    --log-dir "$LOGS/nowords/opinion" --ollama-binary "$OLLAMA_BINARY" --ollama-models "$OLLAMA_MODELS" \
    --verified-dir "$OLLAMA_VERIFIED" "${profile[@]}"
  rc_opinion=$?
  log "nowords: review exit $rc_review, opinion exit $rc_opinion"
  exit $(( rc_review != 0 || rc_opinion != 0 ))
  ;;
review)  # REVIEW through a pinned vLLM server, one GPU per task
  if [ -n "${CUDA_MODULE:-}" ]; then
    module load "$CUDA_MODULE" 2>/dev/null && export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi
  export VLLM_CACHE_ROOT=$WORK/cache/vllm TRITON_CACHE_DIR=$WORK/cache/triton
  export TORCHINDUCTOR_CACHE_DIR=$WORK/cache/inductor FLASHINFER_WORKSPACE_BASE=$WORK/cache/flashinfer
  export SENSELAB_VLLM_LOG_DIR=$LOGS/review/vllm-$I
  drive extend_llm_review "$REVIEW_ROWS/rows_words.jsonl" "${slice[@]}" --config "$REVIEW_CONFIG" --hints "$HINTS" \
    --log-dir "$LOGS/review" --no-refold "${profile[@]}"
  ;;
opinion) # SECOND_OPINION through the pinned Ollama server, one GPU per task
  export OLLAMA_VULKAN=0
  drive extend_second_opinion "$REVIEW_ROWS/rows_words.jsonl" "${slice[@]}" --config "$OPINION_CONFIG" --hints "$HINTS" \
    --log-dir "$LOGS/review/opinion" --ollama-binary "$OLLAMA_BINARY" --ollama-models "$OLLAMA_MODELS" \
    --verified-dir "$OLLAMA_VERIFIED" "${profile[@]}"
  ;;
cohort_collect)    # COHORT's corpus facts per recording (nothing is written into a store)
  drive extend_cohort collect "$MANIFEST" "${slice[@]}" --log-dir "$COHORT" "${profile[@]}"
  ;;
cohort_quantiles)  # the corpus quantiles, over every slice's facts
  want=$COHORT_COLLECT_SLICES
  have=$(ls "$COHORT"/slices/cohort-facts-slice-*-of-"$want".jsonl 2>/dev/null | wc -l)
  [ "$have" -eq "$want" ] || { echo "FATAL: $have of $want cohort fact slices; the quantiles would miss recordings" >&2; exit 2; }
  drive extend_cohort quantiles --facts "$COHORT/slices" --out "$COHORT/cohort_quantiles.json" --commit "$COMMIT" "${profile[@]}"
  ;;
cohort_apply)      # speaker enrollment and COHORT's readings, one participant at a time
  mkdir -p "$COHORT/enrollments" && chmod 700 "$COHORT/enrollments"
  drive extend_cohort apply "$MANIFEST" --quantiles "$COHORT/cohort_quantiles.json" "${slice[@]}" --config "$RUN_CONFIG" \
    --commit "$COMMIT" --log-dir "$LOGS/cohort" --device cpu --enrollments-out "$COHORT/enrollments" "${profile[@]}"
  ;;
cohort_consistency) # read only: the session-against-other-sessions cosines and their lower fence
  drive extend_cohort consistency "$LOGS/cohort" "${profile[@]}" | tee "$COHORT/consistency.txt"
  ;;
refold)      # VERDICT again over every store, reading every pass's readings
  drive extend_refold "$MANIFEST" "${slice[@]}" --hints "$HINTS" --config "$RUN_CONFIG" --log-dir "$LOGS/refold" \
    --commit "$COMMIT" "${profile[@]}"
  ;;
task_audio)  # the plain, enhanced and released redacted streams cut to the task extent
  drive extend_task_audio "$MANIFEST" "${slice[@]}" --config "$RUN_CONFIG" --log-dir "$LOGS/task_audio" "${profile[@]}"
  ;;
task_squim)  # SQUIM over fixed windows of the task extent (a fresh run does not compute it)
  drive extend_task_squim "$MANIFEST" "${slice[@]}" --config "$RUN_CONFIG" --log-dir "$LOGS/task_squim" "${profile[@]}"
  ;;
vectors)     # one compact row per recording, in shards
  umask 077
  mkdir -p "$VECTORS"
  drive triage_recording_vectors "$RUNS" --out "$VECTORS" --slice "$I" --slices "$N" "${profile[@]}"
  ;;
vectors_merge)
  umask 077
  drive triage_recording_vectors --merge "$VECTORS" --out "$VECTORS" "${profile[@]}"
  ;;
table)       # triage.tsv and its column dictionary
  umask 077
  drive triage_table --decisions "$VECTORS/triage_decisions.parquet" --evidence "$VECTORS/triage_evidence.parquet" \
    --run-root "$RUNS" --bids-root "$BIDS_ROOT" --manifest "$MANIFEST" --vectors "$VECTORS/recording_vectors.parquet" \
    --out "$VECTORS/triage.tsv" --threads "${SLURM_CPUS_PER_TASK:-4}" "${profile[@]}"
  ;;
check)       # one decision row per finished run, one commit, one configuration; per-pass status counts
  "$PY" "$CHAIN_DIR/check_run.py" --manifest "$MANIFEST" --decisions "$VECTORS/triage_decisions.parquet" \
    --logs "$LOGS" --commit "$COMMIT"
  ;;
*)
  echo "unknown stage $STAGE" >&2
  exit 2
  ;;
esac

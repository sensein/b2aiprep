# Sourced by step.sh and submit.sh: reads the run's settings file and sets up the environment every step runs in.
# The settings file is $SETTINGS (an absolute path); see settings.example.sh.

CHAIN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${SETTINGS:?SETTINGS must name the settings file of the run}"
# shellcheck source=/dev/null
source "$SETTINGS"

: "${BIDS_ROOT:?BIDS_ROOT is not set in $SETTINGS}"
: "${WORK:?WORK is not set in $SETTINGS}"
: "${PY:?PY is not set in $SETTINGS}"

export PYTHONPATH="${SENSELAB_SRC:+$SENSELAB_SRC:}${B2AIPREP_SRC:+$B2AIPREP_SRC:}${PYTHONPATH:-}"
export HF_HOME HF_HUB_CACHE SENSELAB_CACHE SENSELAB_VENV_CACHE UV_CACHE_DIR
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN
# Online by default: a model configured at a branch (the PII detector, nvidia/gliner-pii@main) is resolved to its
# commit through the Hub API, which offline mode refuses, and the scan then fails for every recording.
if [ "${HF_OFFLINE:-0}" = 1 ]; then export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; fi
export TMPDIR=$WORK/tmp
mkdir -p "$TMPDIR" "$WORK/logs" "$WORK/slurm"

# The senselab drivers, run as modules of the installed (or $SENSELAB_SRC) senselab.
DRIVERS=senselab.audio.workflows.triage.drivers
drive() { "$PY" -u -m "$DRIVERS.$1" "${@:2}"; }

if [ -z "${PROFILE:-}" ]; then
  PROFILE=$("$PY" -c "from importlib.resources import files; print(files('b2aiprep.prepare') / 'resources' / 'triage_profile' / 'adult')")
fi
[ -f "$PROFILE/profile.yaml" ] || { echo "FATAL: no profile.yaml in $PROFILE" >&2; exit 2; }
[ -f "$PROFILE/hints.py" ] || { echo "FATAL: no hints.py in $PROFILE" >&2; exit 2; }
HINTS=$PROFILE   # the profile directory holds the hint builder senselab loads

if [ -z "${COMMIT:-}" ]; then
  if [ -n "${SENSELAB_SRC:-}" ] && git -C "$SENSELAB_SRC" rev-parse HEAD >/dev/null 2>&1; then
    COMMIT=$(git -C "$SENSELAB_SRC" rev-parse HEAD)
  else
    COMMIT=senselab-$("$PY" -c "import importlib.metadata as m; print(m.version('senselab'))")
  fi
fi

RUN_CONFIG=$CHAIN_DIR/configs/release_run_config.yaml
REVIEW_CONFIG=$CHAIN_DIR/configs/review_vllm.yaml
OPINION_CONFIG=$CHAIN_DIR/configs/second_opinion.yaml

# Where each stage reads and writes, under $WORK.
INPUT=$WORK/input.jsonl          # the recordings to triage: one {"source": path} per line
FRESH=$WORK/fresh                # triage_corpus run: out/ (the run roots) and rows/
RUNS=$FRESH/out                  # the run roots every later pass reads and writes in place
MANIFEST=$WORK/runs.jsonl        # the finished runs, with identity columns: what every pass takes
REVIEW_ROWS=$WORK/review_rows    # rows_needs_reread.jsonl, then rows_words.jsonl / rows_nowords.jsonl
COHORT=$WORK/cohort              # cohort facts, quantiles, enrollments
VECTORS=$WORK/recording_vectors  # shards, then the merged parquet tables and triage.tsv
LOGS=$WORK/logs                  # one --log-dir per pass

log() { echo "[$(date -Is)] $*"; }

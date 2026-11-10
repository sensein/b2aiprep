# Settings for one triage run of a Bridge2AI-Voice BIDS tree. Copy this file, fill it in, and pass its path to
# submit.sh: `bash submit.sh /path/to/settings.sh <stage>`. Every step script sources it; nothing else holds a path.
# Shell syntax: it is sourced by bash.

# ---- What is triaged, and where the run writes ------------------------------------------------------------------
BIDS_ROOT=/path/to/bids_tree             # the BIDS tree whose sub-*/ses-*/audio/ recordings are triaged (read only)
WORK=/path/to/scratch/triage_run_YYYY_MM_DD   # everything this run writes: runs, logs, manifests, tables
STEMS_FILE=                              # optional: one file stem per line, to triage only those (a canary)
ALLOW_MISSING=0                          # 1: go on past recordings with no finished run (they are listed, then left out)

# ---- Code -------------------------------------------------------------------------------------------------------
PY=/path/to/python                       # the Python that has senselab (and its triage extras) installed
B2AIPREP_SRC=                            # optional: a b2aiprep checkout's src/, put first on PYTHONPATH
SENSELAB_SRC=                            # optional: a senselab checkout's src/, put first on PYTHONPATH
COMMIT=                                  # recorded on every row and marker; default: senselab's installed version

# The dataset profile (and its hints.py) b2aiprep ships; leave empty for the one in the b2aiprep that $PY imports.
PROFILE=

# ---- Caches (never under $HOME: it is inode-capped) -------------------------------------------------------------
HF_HOME=/path/to/hf_home
HF_HUB_CACHE=/path/to/huggingface/hub    # where the models are already downloaded
SENSELAB_CACHE=$WORK/cache/senselab
SENSELAB_VENV_CACHE=/path/to/senselab-venvs   # senselab's per-model subprocess venvs; reused across runs
UV_CACHE_DIR=/path/to/uv-cache
HF_TOKEN=                                # needed only where a gated model is not in HF_HUB_CACHE yet
HF_OFFLINE=0                             # 1 only when every model the run uses is pinned to a commit SHA: the PII
                                         # detector is configured at a branch, and offline its scan fails silently
FFMPEG_MODULE=ffmpeg/5.1.4               # module loaded for decoding; empty if ffmpeg is already on PATH
CUDA_MODULE=cuda/13.1.0                  # module loaded for the vLLM reviewer

# ---- The reviewer models (GPU) ----------------------------------------------------------------------------------
OLLAMA_BINARY=/path/to/ollama/bin/ollama   # the second opinion's pinned Ollama server
OLLAMA_MODELS=/path/to/ollama/models
OLLAMA_VERIFIED=/path/to/ollama-verified   # where checked model blobs are remembered

# ---- Slurm ------------------------------------------------------------------------------------------------------
CPU_PARTITION=pi_satra,ou_bcs_normal
GPU_PARTITION=ou_bcs_low,pi_satra,mit_preemptable   # every partition with A100s; preempted tasks requeue and resume
GPU_GRES=gpu:a100:1                      # a GPU that holds the 31B reviewer; not a bare gpu:1 (L4/RTX are too small)
LINK_PARTITION=pi_satra,ou_bcs_normal    # the small jobs that submit the next stage
THROTTLE=64                              # at most this many tasks of one array at once (%N); the submit cap is per user
MAIL=                                    # optional: an address for FAIL mail
EXCLUDE_NODES=                           # optional: nodes no stage may use, e.g. one whose filesystem hangs

# Slices per array stage: how many tasks the stage is split into. The values are the released run's.
FRESH_SLICES=240
SESSION_SLICES=128
REPLAY_SLICES=192
NOWORDS_SLICES=16
REVIEW_OPINION_SLICES=51                 # the chain's GPU stage: review, then second opinion, on one GPU per task
REVIEW_SLICES=51                         # the two steps as separate stages, for submitting by hand
OPINION_SLICES=38
COHORT_COLLECT_SLICES=64
COHORT_APPLY_SLICES=128
REFOLD_SLICES=128
TASK_AUDIO_SLICES=64
SQUIM_SLICES=128
VECTORS_SLICES=64

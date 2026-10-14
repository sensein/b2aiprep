"""Prepare senselab triage output as `audio_quality_annotations.tsv` and `consensus_transcripts.parquet`
for a BIDS source tree.

The triage table (`triage.tsv`, one row per recording) is keyed by the file it read. This script:

1. takes each row's ``recording_id`` from the sidecar of the file triage read (``--triage-bids``);
2. checks every row against the tree the release is built from (``--target-bids``): the recording
   must be in its ``recording.tsv`` with the same participant and session, and the task label is
   taken from its file name there (labels can change between builds, e.g. ``high-to-low`` became
   ``glides-high-to-low``); any disagreement stops the run;
3. drops the ``sub-``/``ses-`` prefixes, restores the task timing and recording duration at full
   precision from the triage parquets (the TSV rounds to 4 significant digits), recomputes
   ``task_family``/``branch`` from the current label, and keeps only the release columns;
4. writes the consensus transcript of every recording, one row per word, from each recording's
   ``summary.json`` (the consensus words, their timing and each recogniser's reading);
5. writes the review/discard recordings to ``removal_candidates.tsv`` for the removal list. Nothing is
   removed here: deidentify never reads triage.

Both tables cover every recording. Deidentify keeps the rows of released recordings, and transcripts only
where the recording's audio is released. Copy ``audio_quality_annotations.tsv`` and
``consensus_transcripts.parquet`` to the BIDS root before deidentifying. Their column descriptions are
b2aiprep resources (deidentify writes them); this script stops if a column is not described there or if
triage.json now describes one of Satra's columns differently.

Run with the b2aiprep environment (only pandas and pyarrow are needed):

    python prep_audio_quality_annotations.py --triage-dir .../derivatives/senselab-triage \
        --triage-bids .../bids_adult_2026_09_04 --target-bids .../bids_adult_2026_09_16 --outdir OUT
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

SIDECAR_SUFFIX = "_recording-metadata.json"
AUDIO_SUFFIXES = (".wav", ".flac")

# Columns released, in order. Left out: file paths (source_path, task_audio_*, figure), the pipeline's
# decision and rationale (verdict, release, release_reason, reason, reasons, issues_n, run_status,
# missing, decisive_evidence, annotations) and the per-kind lists empty on released rows
# (interference_in_task, fault_in_task).
RELEASE_COLUMNS = [
    "participant_id", "session_id", "recording_id", "task_name", "task_family", "branch",
    "recording_duration_s", "task_start_s", "task_end_s", "task_duration_s", "task_fraction",
    "task_events_found", "task_events_instructed",
    "level_rel_session_db", "active_s", "snr_db", "noise_floor_dbfs", "session_background_dbfs",
    "speech_over_residual_db", "squim_pesq", "squim_stoi", "squim_si_sdr", "clipped_s", "dropout_s",
    "lexical_words_n", "asr_agreement", "asr_degenerate_n", "masked_s", "masked_fraction",
    "speaker_target_cosine", "speaker_non_task_cosine", "session_consistency_cosine", "enrollment_source",
    "pipeline_commit", "config_hash", "schema_version",
]
# Full-precision sources for the columns the TSV rounds: (parquet, column) -> TSV column.
# task_duration_s and task_fraction are recomputed from these (the parquet's extent_duration_s is itself rounded).
PRECISE = {
    ("triage_decisions.parquet", "extent_start_s"): "task_start_s",
    ("triage_decisions.parquet", "extent_end_s"): "task_end_s",
    ("recording_vectors.parquet", "duration_s"): "recording_duration_s",
}
# Column descriptions live in b2aiprep (prepare/resources/<table>.json); deidentify writes them. This script
# checks them instead: every column described, and the triage.json wording of Satra's columns unchanged.
RESOURCES = Path(__file__).resolve().parents[2] / "src" / "b2aiprep" / "prepare" / "resources"
OWN_COLUMNS = {"participant_id", "session_id", "recording_id", "task_name", "task_family"}  # not triage.json's
# consensus_transcripts.parquet: one row per consensus word, from summary.json
# evidence.consensus_transcript_tokens. Left out: variants (the two readings), sources (which readings are
# filled), onset/offset_spread_s and temporal_uncertainty_s (from the two timings), entity_id, provenance.
RECOGNISERS = {"asr_crisperwhisper": "crisperwhisper", "asr_qwen": "qwen"}
TRANSCRIPT_COLUMNS = [
    "participant_id", "session_id", "recording_id", "task_name", "word_index", "text", "bracketed",
    "start_s", "end_s", "outcome", "agreement",
    *(f"{prefix}_{field}" for prefix in RECOGNISERS.values() for field in ("text", "start_s", "end_s")),
]


def fail(problems, what):
    if problems:
        sys.exit(f"{what}: {len(problems)} problem(s):\n" + "\n".join(str(p) for p in problems[:50]))


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def strip_prefix(value, prefix):
    return value[len(prefix):] if value.startswith(prefix) else value


def lowercase_keys(entry):
    """triage.json entries use Description/Units/Levels; the b2aiprep column descriptions are lowercase."""
    return {k.lower(): v for k, v in entry.items()}


def description_problems(table, columns, triage_info=None):
    """Differences between a table's columns and its b2aiprep description file; with *triage_info*, also
    Satra's columns whose triage.json entry no longer matches the packaged wording."""
    described = json.loads((RESOURCES / f"{table}.json").read_text())
    problems = [f"{c}: not described" for c in columns if c not in described]
    problems += [f"{c}: described but not a column" for c in described if c not in columns]
    if triage_info is not None:
        problems += [f"{c}: triage.json wording changed" for c in columns
                     if c not in OWN_COLUMNS and c in described and lowercase_keys(triage_info[c]) != described[c]]
    return [f"{table}.json: {p}" for p in problems]


def transcript_rows(summary_path, keys):
    """One dict per consensus word of the recording whose summary.json is *summary_path*."""
    tokens = json.loads(Path(summary_path).read_text())["evidence"]["consensus_transcript_tokens"]
    rows = []
    for i, tok in enumerate(tokens):
        readings, timings = tok.get("readings") or {}, tok.get("timings") or {}
        unknown = (set(readings) | set(timings)) - set(RECOGNISERS)
        if unknown:
            raise ValueError(f"{summary_path}: recogniser(s) {sorted(unknown)} not in {sorted(RECOGNISERS)}")
        row = {**keys, "word_index": i, "text": tok["text"], "bracketed": bool(tok["bracketed"]),
               "start_s": tok["timing"]["start_s"], "end_s": tok["timing"]["end_s"],
               "outcome": tok["outcome"], "agreement": tok["agreement"]}
        for model, prefix in RECOGNISERS.items():
            span = timings.get(model) or [None, None]
            row[f"{prefix}_text"] = readings.get(model)
            row[f"{prefix}_start_s"], row[f"{prefix}_end_s"] = span[0], span[1]
        rows.append(row)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--triage-dir", type=Path, required=True, help="the senselab-triage derivatives folder")
    p.add_argument("--triage-bids", type=Path, required=True, help="the BIDS tree triage read (source_path is relative to it)")
    p.add_argument("--target-bids", type=Path, required=True, help="the BIDS tree the release is built from")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--remove-verdicts", default="review,discard", help="verdicts listed in removal_candidates.tsv")
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args(argv)
    a.outdir.mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(a.triage_dir / "triage.tsv", sep="\t", dtype=str, keep_default_na=False)
    column_info = json.loads((a.triage_dir / "triage.json").read_text())
    t["stem"] = t["source_path"].str.rsplit("/", n=1).str[-1].str.replace(r"\.(wav|flac)$", "", regex=True)
    fail(t.index[t["stem"].duplicated()].map(lambda i: t.at[i, "stem"]).tolist(), "duplicate triage stems")

    # 1. recording_id from the sidecar of the file triage read
    sidecar_paths = [a.triage_bids / (sp.rsplit(".", 1)[0] + SIDECAR_SUFFIX) for sp in t["source_path"]]
    with ThreadPoolExecutor(a.workers) as ex:
        sidecars = list(ex.map(read_json, sidecar_paths))
    fail([str(sp) for sp, sc in zip(sidecar_paths, sidecars) if not sc or not sc.get("recording_id")],
         "triage rows without a sidecar recording_id")
    t["recording_id"] = [sc["recording_id"] for sc in sidecars]
    fail(t.loc[t["recording_id"].duplicated(keep=False), "recording_id"].unique().tolist(), "recording_id on two triage rows")

    # 2. check against the target tree; take its task label
    rec = pd.read_csv(a.target_bids / "phenotype" / "task" / "recording.tsv", sep="\t", dtype=str, keep_default_na=False)
    rec_pid = dict(zip(rec["recording_id"], rec["participant_id"].map(lambda v: strip_prefix(v, "sub-"))))
    rec_sid = dict(zip(rec["recording_id"], rec["recording_session_id"].map(lambda v: strip_prefix(v, "ses-"))))
    target = {}
    for sc_path in a.target_bids.glob(f"sub-*/ses-*/audio/*{SIDECAR_SUFFIX}"):
        stem = sc_path.name[: -len(SIDECAR_SUFFIX)]
        if any((sc_path.parent / (stem + s)).exists() for s in AUDIO_SUFFIXES):
            target[stem] = sc_path
    with ThreadPoolExecutor(a.workers) as ex:
        target_ids = dict(zip(ex.map(lambda sp: (read_json(sp) or {}).get("recording_id"), target.values()), target.keys()))
    t["participant_id"] = t["participant_id"].map(lambda v: strip_prefix(v, "sub-"))
    t["session_id"] = t["session_id"].map(lambda v: strip_prefix(v, "ses-"))
    problems = []
    for r in t.itertuples():
        if r.recording_id not in rec_pid:
            problems.append(f"{r.recording_id}: not in target recording.tsv")
        elif (rec_pid[r.recording_id], rec_sid[r.recording_id]) != (r.participant_id, r.session_id):
            problems.append(f"{r.recording_id}: participant/session differ from target recording.tsv")
        elif r.recording_id not in target_ids:
            problems.append(f"{r.recording_id}: no audio file with a sidecar in the target tree")
    fail(problems, "triage rows that do not match the target tree")
    new_task = t["recording_id"].map(lambda rid: target_ids[rid].split("_task-", 1)[1])
    relabelled = int((new_task != t["task"]).sum())
    relabel_pairs = sorted(set(zip(t.loc[new_task != t["task"], "task"], new_task[new_task != t["task"]])))
    t["task"] = t["task_name"] = new_task

    # 3. full precision, task family, release columns. Values are rounded only when written, so the
    # duration and fraction come from unrounded start, end and recording length.
    precise = {}
    for (parquet, col), tsv_col in PRECISE.items():
        values = t["stem"].map(pd.read_parquet(a.triage_dir / parquet, columns=["stem", col]).set_index("stem")[col])
        rounded = pd.to_numeric(t[tsv_col].replace("", None))
        fail(t.loc[(values.isna() != rounded.isna()) | ((values - rounded).abs() > 0.06), "stem"].tolist(),
             f"{parquet}:{col} disagrees with {tsv_col}")
        precise[tsv_col] = values
    precise["task_duration_s"] = precise["task_end_s"] - precise["task_start_s"]
    precise["task_fraction"] = precise["task_duration_s"] / precise["recording_duration_s"]
    for col, values in precise.items():
        t[col] = values.round(3).map(lambda v: "" if pd.isna(v) else f"{v:g}")
    family_branch = t.groupby("task_family")["branch"].agg(lambda s: sorted(set(s)))
    fail([f for f, b in family_branch.items() if len(b) != 1], "task families with more than one branch")
    family_branch = family_branch.map(lambda b: b[0])
    t["task_family"] = t["task"].str.replace(r"(-\d+)+$", "", regex=True)
    fail(sorted(set(t["task_family"]) - set(family_branch.index)), "task families triage has no branch for")
    t["branch"] = t["task_family"].map(family_branch)
    fail([c for c in RELEASE_COLUMNS if c not in t.columns], "release columns missing from triage.tsv")

    fail(description_problems("audio_quality_annotations", RELEASE_COLUMNS, column_info),
         "column descriptions to update in b2aiprep")
    fail(description_problems("consensus_transcripts", TRANSCRIPT_COLUMNS), "column descriptions to update in b2aiprep")
    out = t[RELEASE_COLUMNS]
    out.to_csv(a.outdir / "audio_quality_annotations.tsv", sep="\t", index=False)

    # 4. consensus transcripts, one row per word (summary.json sits next to the summary figure)
    fail(t.loc[t["figure"].eq(""), "stem"].tolist(), "triage rows without a summary figure")
    jobs = [(a.triage_dir / Path(f).with_suffix(".json"), {"participant_id": p, "session_id": s, "recording_id": r, "task_name": k})
            for f, p, s, r, k in zip(t["figure"], t["participant_id"], t["session_id"], t["recording_id"], t["task_name"])]
    with ThreadPoolExecutor(a.workers) as ex:
        words = [row for rows in ex.map(lambda j: transcript_rows(*j), jobs) for row in rows]
    tr = pd.DataFrame(words, columns=TRANSCRIPT_COLUMNS).astype({
        "word_index": "int32", "bracketed": "bool", "agreement": "float64",
        **{c: "float64" for c in TRANSCRIPT_COLUMNS if c.endswith("start_s") or c.endswith("end_s")},
    })
    tr.to_parquet(a.outdir / "consensus_transcripts.parquet", index=False)

    # 5. removal candidates (for the removal list; nothing is removed here)
    remove = [v.strip() for v in a.remove_verdicts.split(",") if v.strip()]
    cand = t.loc[t["verdict"].isin(remove), ["recording_id", "participant_id", "session_id", "task_name", "verdict", "release", "reason", "reasons"]]
    cand.to_csv(a.outdir / "removal_candidates.tsv", sep="\t", index=False)

    report = {
        "triage_dir": str(a.triage_dir), "triage_bids": str(a.triage_bids), "target_bids": str(a.target_bids),
        "pipeline_commit": sorted(set(t["pipeline_commit"])), "config_hash": sorted(set(t["config_hash"])),
        "rows": len(out), "columns": len(RELEASE_COLUMNS), "relabelled_rows": relabelled,
        "relabel_pairs": [list(x) for x in relabel_pairs], "verdicts": t["verdict"].value_counts().to_dict(),
        "removal_candidates": len(cand), "remove_verdicts": remove,
        "transcript_words": len(tr), "transcript_recordings": int(tr["recording_id"].nunique()),
        "transcript_outcomes": tr["outcome"].value_counts().to_dict(),
    }
    (a.outdir / "prep_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "relabel_pairs"}, indent=2))


if __name__ == "__main__":
    main()

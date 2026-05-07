#!/usr/bin/env python3
"""Compare diarization-based and Evans-model unconsented-speaker detection.

Ground truth is derived from a release exclusion list:
  - file stem in exclusion list → ground_truth = "flagged" (positive)
  - file stem not in exclusion list → ground_truth = "clean" (certain negative)

NOTE: "flagged" files were removed from the release for various reasons, not
exclusively unconsented speakers, so precision against positives is approximate.
False-alarm rate on clean files (FP / (FP+TN)) is exact.

Evans model results are joined from a pre-computed predictions CSV (produced by
the model training pipeline). Files are tagged by split:
  - "test"          : in predictions CSV and NOT in train annotations
  - "train"         : in train annotations (model was trained on this file)
  - "not_in_model"  : no Evans prediction available

Evans model performance metrics are computed only on "test" split files to avoid
inflated numbers from train-set memorisation.

Usage
-----
python compare_speaker_detection.py BIDS_DIR EXCLUSION_JSON [options]

Example
-------
python compare_speaker_detection.py \\
    /orcd/data/satra/002/datasets/b2aivoice/post_3.0/v3.1/peds/bids_04_03_26 \\
    /orcd/data/satra/002/datasets/b2aivoice/post_3.0/config/shared_release_config_peds_all/audio_filestems_to_remove.json \\
    --evans-predictions /orcd/data/satra/002/datasets/b2aivoice/b2ai-model/b2ai-models/eval_outputs_adult_uncertainty_all/predictions_with_uncertainty.csv \\
    --evans-train-annotations /orcd/data/satra/002/datasets/b2aivoice/b2ai-model/b2ai-models/annotations/train/peds_annotations_20000.csv \\
    --output results_speaker_detection.tsv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Diarization helpers
# ---------------------------------------------------------------------------

def _parse_diarization(raw: list) -> tuple[int, float, float]:
    """Return (num_distinct_speakers, total_active_speech_s, primary_speaker_ratio).

    primary_speaker_ratio is the fraction of active speech belonging to the
    most-dominant speaker (1.0 when only one speaker is present).
    """
    speaker_durations: dict[str, float] = {}
    for seg in raw:
        if hasattr(seg, "speaker"):
            spk = str(seg.speaker)
            start = float(seg.start)
            end = float(seg.end)
        else:
            spk = str(seg.get("speaker", "speaker_0"))
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        speaker_durations[spk] = speaker_durations.get(spk, 0.0) + max(0.0, end - start)
    total_s = sum(speaker_durations.values())
    if not speaker_durations:
        return 0, 0.0, 1.0
    primary_s = max(speaker_durations.values())
    primary_ratio = primary_s / (total_s + 1e-10)
    return len(speaker_durations), total_s, primary_ratio


def _speaker_category(n: int) -> str:
    if n == 0:
        return "zero"
    if n == 1:
        return "one"
    return "multi"


# ---------------------------------------------------------------------------
# Evans model predictions loader
# ---------------------------------------------------------------------------

def _load_evans_predictions(predictions_csv: str, train_annotations_csv: str | None) -> dict[str, dict]:
    """Load per-file Evans model predictions and label each stem by split.

    Returns a dict mapping stem → {y_pred, confidence, uncertainty, evans_split}.
    """
    import csv as _csv

    train_stems: set[str] = set()
    if train_annotations_csv:
        with open(train_annotations_csv) as fh:
            for row in _csv.DictReader(fh):
                train_stems.add(Path(row["file_path"]).stem)

    records: dict[str, dict] = {}
    with open(predictions_csv) as fh:
        for row in _csv.DictReader(fh):
            stem = Path(row["file_path"]).stem
            split = "train" if stem in train_stems else "test"
            records[stem] = {
                "evans_y_pred":      int(row["y_pred"]),
                "evans_confidence":  float(row["confidence"]),
                "evans_uncertainty": float(row["uncertainty"]),
                "evans_split":       split,
            }

    n_test  = sum(1 for v in records.values() if v["evans_split"] == "test")
    n_train = len(records) - n_test
    print(f"Loaded {len(records):,} Evans predictions  ({n_test:,} test, {n_train:,} train-contaminated).")
    return records


# ---------------------------------------------------------------------------
# Stem → (participant_id, session_id, task_name)
# ---------------------------------------------------------------------------

def _parse_stem(stem: str) -> tuple[str, str, str]:
    """Split 'sub-P_ses-S_task-T' into (P, S, T)."""
    pid = ses = task = ""
    for part in stem.split("_"):
        if part.startswith("sub-"):
            pid = part[4:]
        elif part.startswith("ses-"):
            ses = part[4:]
        elif part.startswith("task-"):
            task = part[5:]
    return pid, ses, task


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _confusion(rows: list[dict], flag_col: str, split_filter: str | None = None) -> dict:
    """Compute TP/FP/TN/FN for a binary flag column.

    When split_filter is set, only rows where evans_split == split_filter are counted.
    Rows where flag_col is None/empty are skipped.
    """
    tp = fp = tn = fn = 0
    for r in rows:
        if split_filter is not None and r.get("evans_split") != split_filter:
            continue
        val = r.get(flag_col)
        if val is None or val == "":
            continue
        gt_pos   = r["ground_truth"] == "flagged"
        pred_pos = int(val) == 1
        if gt_pos and pred_pos:
            tp += 1
        elif not gt_pos and pred_pos:
            fp += 1
        elif not gt_pos and not pred_pos:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else math.nan
    recall    = tp / (tp + fn) if (tp + fn) > 0 else math.nan
    f1 = (2 * precision * recall / (precision + recall)
          if (not math.isnan(precision) and not math.isnan(recall) and (precision + recall) > 0)
          else math.nan)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else math.nan
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "n": tp + fp + tn + fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "false_positive_rate": fpr,
    }


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if not math.isnan(v) else "  nan"
    return str(v)


def _print_summary(rows: list[dict], has_evans: bool) -> None:
    n_total   = len(rows)
    n_clean   = sum(1 for r in rows if r["ground_truth"] == "clean")
    n_flagged = n_total - n_clean

    # Speaker-category breakdown
    cat_counts: dict[str, dict[str, int]] = {}
    for r in rows:
        cat = r["speaker_category"]
        gt  = r["ground_truth"]
        cat_counts.setdefault(cat, {"clean": 0, "flagged": 0})
        cat_counts[cat][gt] += 1

    print("\n" + "=" * 78)
    print("DATASET SUMMARY")
    print("=" * 78)
    print(f"  Total feature files : {n_total:,}")
    print(f"  Ground-truth clean  : {n_clean:,}  (not in exclusion list)")
    print(f"  Ground-truth flagged: {n_flagged:,}  (in exclusion list; reason uncertain)")

    if has_evans:
        for split in ("test", "train", "not_in_model"):
            n = sum(1 for r in rows if r.get("evans_split") == split)
            print(f"    Evans split '{split}': {n:,}")

    print()
    print("DIARIZATION SPEAKER COUNT BREAKDOWN")
    print(f"  {'Category':<8}  {'clean':>8}  {'flagged':>8}  {'total':>8}")
    for cat in ("zero", "one", "multi"):
        counts = cat_counts.get(cat, {"clean": 0, "flagged": 0})
        print(f"  {cat:<8}  {counts['clean']:>8,}  {counts['flagged']:>8,}  {counts['clean']+counts['flagged']:>8,}")

    hdr = (f"  {'Method':<38}  {'N':>7}  {'TP':>6}  {'FP':>6}  {'TN':>6}  {'FN':>6}  "
           f"{'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'FPR':>7}")
    print()
    print("DETECTION PERFORMANCE  (flagged = positive label)")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    def _row(label, col, split_filter=None):
        m = _confusion(rows, col, split_filter=split_filter)
        print(f"  {label:<38}  {m['n']:>7,}  {m['TP']:>6,}  {m['FP']:>6,}  "
              f"{m['TN']:>6,}  {m['FN']:>6,}  "
              f"  {_fmt(m['precision']):>7}  {_fmt(m['recall']):>7}  "
              f"{_fmt(m['F1']):>7}  {_fmt(m['false_positive_rate']):>7}")

    _row("Diarization (multi > 1)  [all files]", "diarization_flag")
    if has_evans:
        _row("Evans model  [test set only]",          "evans_y_pred",  split_filter="test")
        _row("Evans model  [train set — contaminated]","evans_y_pred",  split_filter="train")
        _row("Diarization  [test set only]",           "diarization_flag", split_filter="test")

    print("=" * 78)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("bids_dir",       help="Root BIDS directory containing sub-* folders")
    ap.add_argument("exclusion_json", help="Path to audio_filestems_to_remove.json")
    ap.add_argument(
        "--evans-predictions", metavar="CSV",
        help="Pre-computed predictions CSV from the Evans model training pipeline "
             "(columns: file_path, y_true, y_pred, confidence, uncertainty)",
    )
    ap.add_argument(
        "--evans-train-annotations", metavar="CSV",
        help="Train-split annotation CSV used to flag train-contaminated predictions",
    )
    ap.add_argument("--output", "-o", default="speaker_detection_results.tsv",
                    help="Output TSV path (default: speaker_detection_results.tsv)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N files (0 = all; useful for testing)")
    ap.add_argument("--no-progress", action="store_true",
                    help="Suppress per-file progress output")
    args = ap.parse_args()

    bids_dir = Path(args.bids_dir)
    if not bids_dir.is_dir():
        sys.exit(f"ERROR: BIDS directory not found: {bids_dir}")

    exclusion_path = Path(args.exclusion_json)
    if not exclusion_path.is_file():
        sys.exit(f"ERROR: Exclusion JSON not found: {exclusion_path}")

    with open(exclusion_path) as fh:
        exclusion_set: set[str] = set(json.load(fh))
    print(f"Loaded {len(exclusion_set):,} entries from exclusion list.")

    # Load Evans predictions if provided
    evans_records: dict[str, dict] = {}
    has_evans = bool(args.evans_predictions)
    if has_evans:
        if not Path(args.evans_predictions).is_file():
            sys.exit(f"ERROR: Evans predictions CSV not found: {args.evans_predictions}")
        evans_records = _load_evans_predictions(
            args.evans_predictions,
            args.evans_train_annotations,
        )

    feature_files = sorted(bids_dir.glob("sub-*/ses-*/audio/*_features.pt"))
    if args.limit > 0:
        feature_files = feature_files[: args.limit]
    print(f"Found {len(feature_files):,} feature files to process.")

    fieldnames = [
        "participant_id", "session_id", "task_name", "filestem",
        "ground_truth", "is_speech_task",
        "num_speakers_diarized", "speaker_category", "active_speech_s", "primary_speaker_ratio",
        "diarization_flag",
        "evans_split", "evans_y_pred", "evans_confidence", "evans_uncertainty",
    ]

    rows: list[dict] = []
    output_path = Path(args.output)

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for i, pt_path in enumerate(feature_files):
            stem = pt_path.name.replace("_features.pt", "")
            pid, ses, task = _parse_stem(stem)
            ground_truth = "flagged" if stem in exclusion_set else "clean"

            try:
                feat = torch.load(str(pt_path), weights_only=False, map_location="cpu")
            except Exception as exc:
                print(f"  WARN: could not load {pt_path.name}: {exc}", file=sys.stderr)
                continue

            diarization_raw = feat.get("diarization", [])
            num_spk, active_s, primary_ratio = _parse_diarization(diarization_raw)
            cat = _speaker_category(num_spk)

            ev = evans_records.get(stem, {})
            row = {
                "participant_id":        pid,
                "session_id":            ses,
                "task_name":             task,
                "filestem":              stem,
                "ground_truth":          ground_truth,
                "is_speech_task":        feat.get("is_speech_task", ""),
                "num_speakers_diarized": num_spk,
                "speaker_category":      cat,
                "active_speech_s":       round(active_s, 4),
                "primary_speaker_ratio": round(primary_ratio, 4),
                "diarization_flag":      1 if num_spk > 1 else 0,
                "evans_split":           ev.get("evans_split", "not_in_model"),
                "evans_y_pred":          ev.get("evans_y_pred", ""),
                "evans_confidence":      ev.get("evans_confidence", ""),
                "evans_uncertainty":     ev.get("evans_uncertainty", ""),
            }
            rows.append(row)
            writer.writerow(row)

            if not args.no_progress and (i + 1) % 500 == 0:
                print(f"  processed {i+1:,} / {len(feature_files):,} ...", flush=True)

    print(f"\nResults written to: {output_path}")
    _print_summary(rows, has_evans=has_evans)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write adult and peds participant and recording counts to a TSV.

participants -> unique participant_id across every TSV in <bids>/phenotype
controls     -> two counts, since it is unclear which is the source of truth:
                  control_participants_diagnosis_file: unique participant_id in
                    <bids>/phenotype/diagnosis/control.tsv
                  control_participants_enrollment_flag: unique participant_id with
                    is_control_participant = Yes in <bids>/phenotype/enrollment/eligibility.tsv
                (blank if the file or column does not exist)
recordings   -> .wav / .flac files under each <bids>/sub-*/ folder
hours        -> summed audio durations (samples / sample rate from the file header)
diagnoses    -> adult only, separate file: per phenotype TSV, its gold-standard
                diagnosis column (picked by column name, then by JSON description)

Usage: python bids_participant_summary.py --adult <bids_dir> --peds <bids_dir> [-o OUTPUT]

Either --adult or --peds may be left out. If -o is not given, writes
bids_participant_summary.tsv in the current directory. With --adult, also
writes <output>_diagnoses.tsv: the gold-standard diagnosis column found in each
diagnosis file and its description, for manual review.
"""
import argparse
import csv
import json
import os
import sys
import wave
from concurrent.futures import ThreadPoolExecutor

csv.field_size_limit(sys.maxsize)

AUDIO_EXTENSIONS = (".wav", ".flac")

# Header reads are dominated by network filesystem latency, so read many at once.
AUDIO_READ_THREADS = 32


def tsv_participants(path):
    ids = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "participant_id" not in (reader.fieldnames or []):
            return ids
        for row in reader:
            pid = (row["participant_id"] or "").strip()
            if pid:
                ids.add(pid[4:] if pid.startswith("sub-") else pid)
    return ids


def phenotype_participants(bids_dir):
    ids = set()
    for root, _, files in os.walk(os.path.join(bids_dir, "phenotype")):
        for name in files:
            if name.endswith(".tsv"):
                ids |= tsv_participants(os.path.join(root, name))
    return ids


def control_participants_diagnosis_file(bids_dir):
    path = os.path.join(bids_dir, "phenotype", "diagnosis", "control.tsv")
    return len(tsv_participants(path)) if os.path.isfile(path) else ""


def control_participants_enrollment_flag(bids_dir):
    path = os.path.join(bids_dir, "phenotype", "enrollment", "eligibility.tsv")
    if not os.path.isfile(path):
        return ""
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "is_control_participant" not in (reader.fieldnames or []):
            return ""
        return len({row["participant_id"].strip() for row in reader
                    if (row["is_control_participant"] or "").strip().lower() == "yes"})


def column_descriptions(tsv_path):
    """Column -> description from the TSV's JSON sidecar, if there is one."""
    try:
        with open(os.path.splitext(tsv_path)[0] + ".json") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return {}
    descriptions = {}
    for table in meta.values():
        if isinstance(table, dict):
            for col, info in (table.get("data_elements") or {}).items():
                if isinstance(info, dict):
                    descriptions[col] = info.get("description") or ""
    return descriptions


def gold_standard_column(columns, descriptions):
    # "___" columns are checkbox sub-fields (e.g. confirmation method), not the diagnosis.
    columns = [c for c in columns if "___" not in c]
    tiers = [
        [c for c in columns if c.endswith("gold_standard_diagnosis")],
        [c for c in columns if "gold_standard" in c],
        [c for c in columns if c.endswith("_gs") or "_gs_" in c],
        [c for c in columns if "gold standard" in descriptions.get(c, "").lower()],
    ]
    for candidates in tiers:
        if candidates:
            return min(candidates, key=len)
    return None


def diagnoses(bids_dir):
    """Gold-standard column per diagnosis TSV (any TSV under a diagnosis folder or with one)."""
    found = []
    for root, _, files in os.walk(os.path.join(bids_dir, "phenotype")):
        for name in sorted(files):
            # control.tsv is the control group, counted in the main summary instead.
            if not name.endswith(".tsv") or name == "control.tsv":
                continue
            path = os.path.join(root, name)
            with open(path, newline="") as f:
                columns = next(csv.reader(f, delimiter="\t"), [])
            descriptions = column_descriptions(path)
            gs_col = gold_standard_column(columns, descriptions)
            if gs_col is None and os.path.basename(root) != "diagnosis":
                continue
            found.append({
                "diagnosis": os.path.splitext(name)[0],
                "gs_column": gs_col or "",
                "source_file": os.path.relpath(path, bids_dir),
                "description": descriptions.get(gs_col, "") if gs_col else "",
            })
    return found


def wav_seconds(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def flac_seconds(path):
    # STREAMINFO is always the first metadata block, right after the "fLaC" marker.
    with open(path, "rb") as f:
        head = f.read(42)
    if len(head) < 42 or head[:4] != b"fLaC" or head[4] & 0x7F != 0:
        raise ValueError("no FLAC STREAMINFO block")
    info = int.from_bytes(head[18:26], "big")
    sample_rate = info >> 44
    total_samples = info & 0xFFFFFFFFF
    if not sample_rate or not total_samples:
        raise ValueError("FLAC header has no sample rate or sample count")
    return total_samples / sample_rate


def audio_seconds(path):
    try:
        if path.endswith(".flac"):
            return flac_seconds(path)
        return wav_seconds(path)
    except (wave.Error, EOFError, OSError, ValueError):
        return None


def recordings_per_participant(bids_dir):
    counts = {}
    audio_paths = []
    for entry in os.scandir(bids_dir):
        if entry.is_dir() and entry.name.startswith("sub-"):
            n = 0
            for root, _, files in os.walk(entry.path):
                for name in files:
                    if name.endswith(AUDIO_EXTENSIONS):
                        n += 1
                        audio_paths.append(os.path.join(root, name))
            counts[entry.name[4:]] = n

    with ThreadPoolExecutor(AUDIO_READ_THREADS) as pool:
        durations = list(pool.map(audio_seconds, audio_paths))
    seconds = sum(d for d in durations if d is not None)
    unreadable = sum(1 for d in durations if d is None)
    return counts, seconds, unreadable


def summarize(bids_dir):
    if not os.path.isdir(os.path.join(bids_dir, "phenotype")):
        sys.exit(f"No phenotype folder in {bids_dir}")

    pheno_ids = phenotype_participants(bids_dir)
    counts, seconds, unreadable = recordings_per_participant(bids_dir)
    total = sum(counts.values())
    with_recordings = sum(1 for n in counts.values() if n > 0)

    return [
        ("participants_in_phenotype", len(pheno_ids)),
        ("control_participants_diagnosis_file", control_participants_diagnosis_file(bids_dir)),
        ("control_participants_enrollment_flag", control_participants_enrollment_flag(bids_dir)),
        ("participant_folders", len(counts)),
        ("participants_with_recordings", with_recordings),
        ("total_recordings", total),
        ("avg_recordings_per_participant", round(total / len(counts)) if counts else 0),
        ("avg_recordings_per_participant_with_recordings",
         round(total / with_recordings) if with_recordings else 0),
        ("recording_hours", f"{seconds / 3600:.2f}"),
        ("unreadable_audio_files", unreadable),
    ]


def write_diagnoses(bids_dir, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["diagnosis", "gs_column", "source_file", "description"])
        for d in diagnoses(bids_dir):
            writer.writerow([d["diagnosis"], d["gs_column"] or "<none found>",
                             d["source_file"], d["description"]])
    print(f"Wrote {path}")


def main(cohorts, out_path):
    results = {}
    for name, bids_dir in cohorts.items():
        print(f"Summarizing {name}: {bids_dir}")
        results[name] = dict(summarize(bids_dir))

    metrics = list(next(iter(results.values())))
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["metric"] + list(results))
        for metric in metrics:
            writer.writerow([metric] + [results[name][metric] for name in results])
    print(f"Wrote {out_path}")

    # Diagnoses are adult only; peds has no gold-standard diagnosis columns.
    if "adult" in cohorts:
        write_diagnoses(cohorts["adult"], os.path.splitext(out_path)[0] + "_diagnoses.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adult and peds participant and recording counts.")
    parser.add_argument("--adult", help="adult BIDS directory (contains phenotype/ and sub-*/)")
    parser.add_argument("--peds", help="peds BIDS directory (contains phenotype/ and sub-*/)")
    parser.add_argument("-o", "--output", default="bids_participant_summary.tsv",
                        help="output TSV path (default: bids_participant_summary.tsv in the current directory)")
    args = parser.parse_args()

    cohorts = {name: path for name, path in (("adult", args.adult), ("peds", args.peds)) if path}
    if not cohorts:
        parser.error("give at least one of --adult or --peds")
    main(cohorts, args.output)

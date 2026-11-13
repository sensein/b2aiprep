"""Check a finished triage chain before its tables are used: every pass's status counts, and one decision per run.

    python check_run.py --manifest RUNS.jsonl --decisions triage_decisions.parquet --logs LOG_DIR [--commit SHA]

- Per pass (each folder under ``LOG_DIR`` with a ``slices/`` log): how many rows ended in each status, and the stems of
  the ``error`` / ``errored`` / ``failed`` rows, written to ``LOG_DIR/<pass>.bad_stems.txt``.
- The decision table: one row per manifest run and no other, no stem twice, one ``config_hash``; with ``--commit``, every
  row's ``commit`` is that one. Counts of ``verdict``, ``release`` and ``reason`` are printed.

Prints stems and counts only. Exit 0 when the table covers the manifest exactly and holds one commit and one
configuration, 3 otherwise; a pass with error rows is reported but does not change the exit code (the pass itself
failed its job for them).
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pyarrow.parquet as pq

BAD = ("error", "errored", "failed")


def pass_counts(logs: Path) -> dict[str, dict[str, int]]:
    """Each pass's row statuses from its slice logs; writes each pass's bad stems beside them."""
    out: dict[str, dict[str, int]] = {}
    for slices in sorted(logs.rglob("slices")):
        if not slices.is_dir():
            continue
        name = str(slices.parent.relative_to(logs)).replace("/", ".")
        counts: collections.Counter[str] = collections.Counter()
        bad: list[str] = []
        for path in sorted(slices.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.startswith("{"):
                    continue
                row = json.loads(line)
                if "stem" not in row:
                    continue
                status = str(row.get("status") or ("ok" if row.get("ok") else "unrecorded"))
                counts[status] += 1
                if status.split(":", 1)[0] in BAD:
                    bad.append(str(row["stem"]))
        out[name] = dict(sorted(counts.items()))
        bad_file = logs / f"{name}.bad_stems.txt"
        if bad:
            bad_file.write_text("\n".join(bad) + "\n", encoding="utf-8")
        elif bad_file.exists():
            bad_file.unlink()
    return out


def coverage(
    manifest: Path, decisions: Path, commit: str | None
) -> tuple[list[str], dict[str, object]]:
    """What is wrong with the decision table against the manifest, and its summary counts."""
    names = set(pq.read_schema(decisions).names)
    wanted = [
        c for c in ("stem", "commit", "config_hash", "verdict", "release", "reason") if c in names
    ]
    table = pq.read_table(decisions, columns=wanted).to_pydict()
    stems = [str(s) for s in table["stem"]]
    runs = [
        json.loads(line)["stem"]
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    problems: list[str] = []
    if len(set(stems)) != len(stems):
        problems.append(f"{len(stems) - len(set(stems))} repeated stems in the decision table")
    if missing := sorted(set(runs) - set(stems)):
        problems.append(f"{len(missing)} manifest runs with no decision row, e.g. {missing[:3]}")
    if extra := sorted(set(stems) - set(runs)):
        problems.append(f"{len(extra)} decision rows not in the manifest, e.g. {extra[:3]}")
    hashes = sorted({str(h) for h in table.get("config_hash", [])})
    if len(hashes) != 1:
        problems.append(f"{len(hashes)} configuration hashes: {hashes[:5]}")
    commits = sorted({str(c) for c in table.get("commit", [])})
    if commit is not None and commits != [commit]:
        problems.append(f"commits {commits[:5]}, expected {commit}")
    summary: dict[str, object] = {
        "rows": len(stems),
        "manifest": len(runs),
        "config_hash": hashes,
        "commit": commits,
    }
    for column in ("verdict", "release", "reason"):
        if column in table:
            summary[column] = collections.Counter(table[column]).most_common(20)
    return problems, summary


def main(argv: list[str] | None = None) -> int:
    """Print every pass's counts and the coverage verdict."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--commit", default=None)
    args = parser.parse_args(argv)
    for name, counts in pass_counts(args.logs).items():
        print(f"{name}: {counts}")
    problems, summary = coverage(args.manifest, args.decisions, args.commit)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("coverage", "OK" if not problems else "FAILED: " + "; ".join(problems))
    return 3 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

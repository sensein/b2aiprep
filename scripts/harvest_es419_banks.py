#!/usr/bin/env python3
"""Harvest the Spanish (es-419) stimulus banks from the bridge2ai-redcap docs.

Parses the committed `es-419` acoustic-task docs (no manual transcription, so the
WER reference strings are exactly the project's) into stimulus banks that mirror
the English bank shapes, plus a small static-inline Spanish bank for the passage /
recall reference tasks. Spanish sessions received only the current protocol, so
only current-protocol tasks are harvested.

Outputs (into src/b2aiprep/prepare/resources/task_registry/):
  harvard_sentences_bank_es_419.json   {lists: {"<L>": [10 strings]}}
  cape_v_sentences_bank_es_419.json    {lists: {"v2": {"<N>": str}}}
  static_stimulus_es_419.json          {"<task_id>": {stimulus_text, instructions}}

Usage:
  python scripts/harvest_es419_banks.py --redcap-root /path/to/bridge2ai-redcap
  python scripts/harvest_es419_banks.py --redcap-root ... --check   # fail on drift
"""
import argparse
import json
import re
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "src" / "b2aiprep" / "prepare" / "resources" / "task_registry"
CAPE_V_INSTRUCTION = "Por favor, lea las siguientes frases en voz alta con su voz habitual."


def _acoustic(root: Path) -> Path:
    return root / "docs" / "Adults" / "Acoustic Tasks" / "Current"


def harvest_harvard(root: Path) -> dict:
    text = (_acoustic(root) / "Harvard Sentences" / "HarvardSentences-es-419.md").read_text()
    lists: dict = {}
    current = None
    for line in text.splitlines():
        h = re.match(r"^##\s*List\s*(\d+)\s*$", line.strip())
        if h:
            current = h.group(1)
            lists[current] = []
            continue
        item = re.match(r"^\d+\.\s+(.*\S)\s*$", line.strip())
        if item and current is not None:
            lists[current].append(item.group(1).strip())
    return {
        "description": "Harvard/IEEE sentences, Spanish (es-419) set, from "
        "bridge2ai-redcap Current/Harvard Sentences/HarvardSentences-es-419.md.",
        "lists": lists,
    }


def _recording_blocks(text: str):
    """Yield (recording_number, [non-badge text lines]) for each '![recording_N]' block."""
    blocks = re.split(r"\n-{3,}\n", text)
    for block in blocks:
        m = re.search(r"\[recording_(\d+)\]\[recording_\d+\]", block)
        if not m:
            continue
        lines = []
        for ln in block.splitlines():
            s = ln.strip()
            if not s or s.startswith("![") or s.startswith("[") or s.startswith("#") or s.startswith(">"):
                continue
            lines.append(s)
        yield m.group(1), lines


def harvest_cape_v(root: Path) -> dict:
    p = _acoustic(root) / "Cape V Sentences - v2" / "Cape V Sentences - Acoustic Task Description (Spanish).md"
    v2: dict = {}
    for num, lines in _recording_blocks(p.read_text()):
        sentences = [l for l in lines if l != CAPE_V_INSTRUCTION]
        if sentences:
            v2[num] = sentences[-1]
    return {
        "description": "CAPE-V sentences, Spanish (es-419), current protocol (v2), from "
        "bridge2ai-redcap Cape V Sentences - v2 (Spanish) description.",
        "lists": {"v2": v2},
    }


def _blockquote(text: str) -> str:
    quoted = [ln.strip()[1:].strip() for ln in text.splitlines() if ln.strip().startswith(">")]
    return " ".join(q for q in quoted if q).strip()


def _instruction_lines(text: str, stop_prefixes=(">",)) -> str:
    """Prose instruction lines after the first recording badge, before the stimulus."""
    out = []
    seen_badge = False
    for ln in text.splitlines():
        s = ln.strip()
        if "[recording_1]" in s:
            seen_badge = True
            continue
        if not seen_badge:
            continue
        if not s or s.startswith("![") or s.startswith("["):
            continue
        if any(s.startswith(p) for p in stop_prefixes) or re.match(r"^\d+\.", s):
            break
        out.append(s)
    return " ".join(out).strip()


def harvest_static(root: Path) -> dict:
    a = _acoustic(root)
    out: dict = {}

    cat = (a / "Caterpillar Passage" / "Caterpillar Passage - Acoustic Task Description (Spanish).md").read_text()
    out["adult.caterpillar-passage"] = {
        "stimulus_text": _blockquote(cat),
        "instructions": _instruction_lines(cat),
        "speech_type": "read",
    }

    story = (a / "Story Recall - v2" / "Story Recall - Acoustic Task Description (Spanish).md").read_text()
    scenes = [m.group(1).strip() for m in re.finditer(r"^\d+\.\s+(.*\S)\s*$", story, re.MULTILINE)]
    # The instruction block is the prose before the first numbered scene.
    instr = []
    for ln in story.splitlines():
        s = ln.strip()
        if re.match(r"^\d+\.", s):
            break
        if s and not s.startswith(("#", "!", "[", "<", ">")) and "Voice as a Biomarker" not in s:
            instr.append(s)
    out["adult.story-recall.v2"] = {
        "stimulus_text": " ".join(scenes).strip(),
        "instructions": " ".join(instr).strip(),
        "speech_type": "recall",
    }
    return out


BANKS = {
    "harvard_sentences_bank_es_419.json": harvest_harvard,
    "cape_v_sentences_bank_es_419.json": harvest_cape_v,
    "static_stimulus_es_419.json": harvest_static,
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--redcap-root", required=True, type=Path)
    ap.add_argument("--check", action="store_true", help="fail if committed output would change")
    args = ap.parse_args(argv)

    drift = False
    for fname, fn in BANKS.items():
        obj = fn(args.redcap_root)
        new = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
        path = RES / fname
        if args.check:
            old = path.read_text() if path.exists() else ""
            if old != new:
                drift = True
                print(f"DRIFT: {fname} would change")
        else:
            path.write_text(new)
            n = len(obj.get("lists", obj))
            print(f"wrote {fname} ({n} groups)")
    if args.check and drift:
        sys.exit(1)


if __name__ == "__main__":
    main()

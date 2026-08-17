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


# The stimulus text (WER reference) is keyed by task FAMILY so any version of a
# family that a Spanish session recorded (incl. a v1-labelled recording) picks it
# up -- Spanish only has the current stimulus set per family.
def harvest_static(root: Path) -> dict:
    a = _acoustic(root)
    out: dict = {}

    cat = (a / "Caterpillar Passage" / "Caterpillar Passage - Acoustic Task Description (Spanish).md").read_text()
    out["caterpillar-passage"] = {"stimulus_text": _blockquote(cat), "speech_type": "read"}

    story = (a / "Story Recall - v2" / "Story Recall - Acoustic Task Description (Spanish).md").read_text()
    scenes = [m.group(1).strip() for m in re.finditer(r"^\d+\.\s+(.*\S)\s*$", story, re.MULTILINE)]
    out["story-recall"] = {"stimulus_text": " ".join(scenes).strip(), "speech_type": "recall"}
    return out


# Family slug -> Spanish "Acoustic Task Description" file (relative to Current/),
# for the current-protocol adult tasks Spanish sessions record.
FAMILY_DESC = {
    "harvard-sentences": "Harvard Sentences/Harvard Sentences - Acoustic Task Description (Spanish).md",
    "cape-v-sentences": "Cape V Sentences - v2/Cape V Sentences - Acoustic Task Description (Spanish).md",
    "caterpillar-passage": "Caterpillar Passage/Caterpillar Passage - Acoustic Task Description (Spanish).md",
    "story-recall": "Story Recall - v2/Story Recall - Acoustic Task Description (Spanish).md",
    "free-speech": "Free Speech - v2/Free Speech - Acoustic Task Description (Spanish).md",
    "glides": "Glides/Glides - Acoustic Task Description (Spanish).md",
    "diadochokinesis": "Diadochokinesis - v2/Diadochokinesis - Acoustic Task Description (Spanish).md",
    "maximum-phonation-time": "Maximum Phonation Time - v2/Maximum Phonation Time - Acoustic Task Description (Spanish).md",
    "prolonged-vowel": "Prolonged Vowel/Prolonged Vowel - Acoustic Task Description (Spanish).md",
    "loudness": "Loudness - v2/Loudness - Acoustic Task Description (Spanish).md",
    "picture-description": "Picture Description/Picture Description - Acoustic Task Description (Spanish).md",
    "respiration-and-cough": "Respiration and cough - v2/Respiration and cough - Acoustic Task Description (Spanish).md",
    # Questionnaire-join tasks: no current Spanish participant recorded these, but
    # they have Spanish descriptions, so harvest their instructions too for
    # completeness (their per-participant stimulus comes from the join, which is
    # already language-agnostic).
    "productive-vocabulary": "Productive Vocabulary/Productive Vocabulary - Acoustic Task Description (Spanish).md",
    "random-item-generation": "Random Item Generation - v2/Random Item Generation - Acoustic Task Description (Spanish).md",
    "word-color-stroop": "Word-color Stroop/Word-color Stroop - Acoustic Task Description (Spanish).md",
}

# Prose lines that are never instruction text (logo/nav/preamble boilerplate).
_SKIP_PROSE = ("Voice as a Biomarker", "Para ver la lista completa")


def _instruction_from_desc(text: str) -> str:
    """Extract the Spanish instruction from an Acoustic Task Description page.

    The instruction is the imperative prose that recurs across the recording
    blocks; the stimulus (a read sentence, a scene) appears once. So: keep the
    prose lines that occur more than once (in first-seen order). If nothing
    repeats (a single-block passage), keep the prose that isn't the blockquote /
    numbered-scene stimulus -- i.e. the preamble instruction.
    """
    prose = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith(("#", "!", "[", "<", ">", "|", "-")):
            continue
        if re.match(r"^\d+\.\s", s):  # numbered stimulus scene
            continue
        if any(k in s for k in _SKIP_PROSE):
            continue
        prose.append(s)
    counts: dict = {}
    for s in prose:
        counts[s] = counts.get(s, 0) + 1
    repeated = [s for s in dict.fromkeys(prose) if counts[s] > 1]
    lines = repeated if repeated else list(dict.fromkeys(prose))
    return " ".join(lines).strip()


def harvest_instructions(root: Path) -> dict:
    a = _acoustic(root)
    out: dict = {}
    for family, rel in FAMILY_DESC.items():
        p = a / rel
        if p.exists():
            instr = _instruction_from_desc(p.read_text())
            if instr:
                out[family] = instr
    return out


BANKS = {
    "harvard_sentences_bank_es_419.json": harvest_harvard,
    "cape_v_sentences_bank_es_419.json": harvest_cape_v,
    "static_stimulus_es_419.json": harvest_static,
    "task_instructions_es_419.json": harvest_instructions,
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

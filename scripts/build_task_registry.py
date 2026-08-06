#!/usr/bin/env python3
"""Generate the b2aiprep audio-task registry from a bridge2ai-redcap checkout.

Reads the redcap docs tree (the GFM index tables, the per-recording tables, and
the plain-text stimulus files) plus the existing, human-curated flat
``audio_task_descriptions.json`` (for instructions and already-transcribed
image-task text), and writes a hierarchy-mirroring registry + stimulus banks to
``src/b2aiprep/prepare/resources/task_registry/``.

Dev/CI only, stdlib only. The output is *vendored* (committed) exactly like
``redcap2rs`` -- there is no runtime or submodule dependency on bridge2ai-redcap.

Usage:
    python scripts/build_task_registry.py --redcap-root /path/to/bridge2ai-redcap
    python scripts/build_task_registry.py --redcap-root ... --check   # CI drift guard
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCES = REPO_ROOT / "src" / "b2aiprep" / "prepare" / "resources"
REGISTRY_DIR = RESOURCES / "task_registry"
FLAT_DESCRIPTIONS = RESOURCES / "audio_task_descriptions.json"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def slug(text: str) -> str:
    """Lowercase, alphanumeric-with-hyphens slug (parentheses/spaces -> '-')."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def parse_gfm_tables(md_text: str):
    """Parse GitHub-flavored markdown tables; return list of (header, rows)."""
    tables = []
    header = None
    rows = None
    for line in md_text.splitlines():
        s = line.strip()
        is_row = s.startswith("|") and s.endswith("|")
        if is_row:
            cells = [c.strip() for c in s.strip("|").split("|")]
            if header is None:
                header = cells
                rows = []
                continue
            if all(set(c) <= set(":- ") and "-" in c for c in cells):
                continue  # separator row
            if len(cells) == len(header):
                rows.append(OrderedDict(zip(header, cells)))
        else:
            if header is not None:
                tables.append((header, rows))
            header, rows = None, None
    if header is not None:
        tables.append((header, rows))
    return tables


def find_key(header, *needles):
    """First header cell containing all needles (case-insensitive)."""
    for h in header:
        hl = h.lower()
        if all(n.lower() in hl for n in needles):
            return h
    return None


def strip_version(name: str):
    """Return (family, version) splitting a trailing '(vN)' token."""
    m = re.search(r"\(v(\d+)\)", name)
    version = f"v{m.group(1)}" if m else None
    family = re.sub(r"\s*\(v\d+\)\s*", " ", name).strip()
    return family, version


def clean_recording_name(name: str) -> str:
    """Drop markdown emphasis annotations like '*(second attempt)*'."""
    name = re.sub(r"\*\([^)]*\)\*", "", name)
    return re.sub(r"\s+", " ", name).strip()


# --------------------------------------------------------------------------- #
# Stimulus-bank extraction
# --------------------------------------------------------------------------- #
def extract_harvard_bank(redcap_root: Path) -> dict:
    path = redcap_root / "docs/Adults/Acoustic Tasks/Current/Harvard Sentences/HarvardSentences.md"
    lists: "OrderedDict[str, list]" = OrderedDict()
    current = None
    for line in path.read_text().splitlines():
        m = re.match(r"##\s*List\s*(\d+)", line.strip())
        if m:
            current = m.group(1)
            lists[current] = []
            continue
        m = re.match(r"(\d+)\.\s+(.*\S)", line.strip())
        if m and current is not None:
            lists[current].append(m.group(2).strip())
    return {"description": f"Harvard/IEEE sentences, {len(lists)} lists x 10, from bridge2ai-redcap "
                           "docs/Adults/Acoustic Tasks/Current/Harvard Sentences.",
            "lists": lists}


def extract_productive_vocabulary_bank(redcap_root: Path) -> dict:
    path = (redcap_root / "docs/Adults/Acoustic Tasks/Current/Productive Vocabulary/"
            "Productive Vocabulary Words.md")
    lists: "OrderedDict[str, list]" = OrderedDict()
    current = None
    for line in path.read_text().splitlines():
        m = re.match(r"###\s*List\s*(\d+)", line.strip())
        if m:
            current = m.group(1)
            lists[current] = []
            continue
        m = re.match(r"-\s+(.*\S)", line.strip())
        if m and current is not None:
            lists[current].append(m.group(1).strip())
    return {"description": "Productive Vocabulary word pool (per-participant 6 words are randomized "
                           "from these lists; source bridge2ai-redcap Productive Vocabulary Words.",
            "lists": lists}


def extract_random_categories(redcap_root: Path) -> dict:
    base = redcap_root / "docs/Adults/Acoustic Tasks/Current/Random Item Generation - v2"
    cats: "OrderedDict[str, list]" = OrderedDict()
    for cat_file in sorted(base.glob("Random Item Generation Category_*.md")):
        if "es-419" in cat_file.name:
            continue
        key = re.search(r"Category_(\d+)", cat_file.name).group(1)
        items = []
        for line in cat_file.read_text().splitlines():
            m = re.match(r"-\s+(.*\S)", line.strip())
            if m:
                items.append(re.sub(r"[“”]", '"', m.group(1).strip()))
        cats[f"category_{key}"] = items
    return {"description": "Random Item Generation category options (source bridge2ai-redcap RIG v2).",
            "categories": cats}


def extract_cape_v_bank() -> dict:
    """CAPE-V v1/v2 sentences harvested from the curated flat descriptions file."""
    flat = json.loads(FLAT_DESCRIPTIONS.read_text(), object_pairs_hook=OrderedDict)
    bank = {"v1": OrderedDict(), "v2": OrderedDict()}
    for key, entry in flat.items():
        m = re.match(r"cape-V-sentences-(\(v2\)-)?(\d+)$", key)
        if not m:
            continue
        version = "v2" if m.group(1) else "v1"
        prompts = entry.get("prompts", [])
        if prompts:
            bank[version][m.group(2)] = prompts[0]
    return {"description": "CAPE-V sentences (v1 retired / v2 current) harvested from the curated "
                           "audio_task_descriptions.json.",
            "lists": bank}


# --------------------------------------------------------------------------- #
# Prompt-source policy: the single home for human knowledge about how each task
# family's per-recording stimulus is resolved. Keyed by family slug.
# --------------------------------------------------------------------------- #
POLICY = {
    # slug: (speech_type, prompt_ref)
    "harvard-sentences": ("read", {"type": "stimulus-bank", "bank": "harvard_sentences_bank",
                                    "select": "list-index"}),
    "cape-v-sentences": ("read", {"type": "stimulus-bank", "bank": "cape_v_sentences_bank",
                                   "select": "version-index"}),
    "repeating-words": ("read", {"type": "stimulus-bank", "bank": "repeating_words_bank",
                                  "select": "index-or-word"}),
    "productive-vocabulary": ("elicited", {"type": "questionnaire-join", "instrument": "vocab",
                                            "join_key": "vocabulary_recording_acoustic_task_id",
                                            "select": "index-field",
                                            "field_template": "vocabulary_item_word_{i}"}),
    "random-item-generation": ("elicited", {"type": "questionnaire-join", "instrument": "random",
                                             "join_key": "random_recording_acoustic_task_id",
                                             "select": "field",
                                             "field": "random_item_generation_category"}),
    "word-color-stroop": ("read", {"type": "questionnaire-join", "instrument": "stroop",
                                    "join_key": "stroop_recording_acoustic_task_id",
                                    "select": "color-list", "n": 15,
                                    "field_template": "stroop_item_color_{i}"}),
    "story-recall": ("recall", {"type": "static-inline"}),
    "cinderella-story": ("recall", {"type": "static-inline"}),
    "rainbow-passage": ("read", {"type": "static-inline"}),
    "caterpillar-passage": ("read", {"type": "static-inline"}),
    "reading-passage": ("read", {"type": "image", "note": "passages presented as images"}),
    "repeating-sentences": ("read", {"type": "image", "note": "sentences presented as images"}),
    "identifying-pictures": ("elicited", {"type": "image"}),
    "picture-description": ("elicited", {"type": "image"}),
    "noisy-sounds": ("non-lexical", {"type": "image"}),
    "silly-sounds": ("non-lexical", {"type": "image"}),
    "long-sounds": ("non-lexical", {"type": "image"}),
    "diadochokinesis": ("non-lexical", {"type": "static-inline"}),
    "prolonged-vowel": ("non-lexical", {"type": "static-inline"}),
    "glides": ("non-lexical", {"type": "static-inline"}),
    "loudness": ("non-lexical", {"type": "static-inline"}),
    "maximum-phonation-time": ("non-lexical", {"type": "static-inline"}),
    "respiration-and-cough": ("non-lexical", {"type": "static-inline"}),
    "voluntary-cough": ("non-lexical", {"type": "static-inline"}),
    "breath-sounds": ("non-lexical", {"type": "static-inline"}),
    # elicited / spontaneous-or-recited families (no provided target text)
    "free-speech": ("elicited", {"type": "static-inline"}),
    "free-speech-voice": ("elicited", {"type": "static-inline"}),
    "animal-fluency": ("elicited", {"type": "static-inline"}),
    "generative-naming-task": ("elicited", {"type": "static-inline"}),
    "open-response-questions": ("elicited", {"type": "static-inline"}),
    "conversation-2-to-4": ("elicited", {"type": "static-inline"}),
    "conversation-4-to-6": ("elicited", {"type": "static-inline"}),
    "conversation-6-plus": ("elicited", {"type": "static-inline"}),
    "abcs-and-123s": ("elicited", {"type": "static-inline"}),
    "days-and-number-naming": ("elicited", {"type": "static-inline"}),
    "role-naming-tasks-sounds": ("elicited", {"type": "static-inline"}),
}
DEFAULT_POLICY = ("elicited", {"type": "static-inline"})

# Families whose per-recording stimulus is resolved dynamically at BIDS build via
# the task-level prompt_ref (bank or questionnaire) -- do NOT enumerate their
# recordings from Recordings.md (Harvard uses "[List #]" placeholders; counts are
# unreliable). All other tasks get nested recording entries.
BANKED_FAMILIES = {
    "harvard-sentences", "repeating-words", "cape-v-sentences", "productive-vocabulary",
    "random-item-generation", "word-color-stroop",
}

# Recording sub-name (slug) -> flat-file key, for the few sub-recordings whose name
# doesn't slug-match a flat key directly. Grouped tasks (Conversation, Generative
# Naming, Days/Number, Role naming) each get PER-RECORDING instructions harvested
# from these keys -- there is no single task-level instruction for them.
SUBTASK_TO_FLAT = {
    "animals": "naming-animals",
    "food": "naming-food",
    "numbers": "123s",
}

# Task-level bridge for renamed-but-UNIFORM tasks: the redcap family name differs
# from the curated flat key, but the instruction is a single one shared by all the
# task's recordings (unlike grouped tasks, so a single mapping is accurate here).
FAMILY_TO_FLAT = {
    "identifying-pictures": "picture",
    "reading-passage": "passage",
    "repeating-sentences": "sentence",
}


# --------------------------------------------------------------------------- #
# Task-index parsing
# --------------------------------------------------------------------------- #
def parse_task_index(redcap_root: Path, population: str):
    """Yield task dicts from an Acoustic Tasks index (Active + Retired sections)."""
    fname = "Adult Acoustic Tasks.md" if population == "adult" else "Pediatric Acoustic Tasks.md"
    sub = "Adults" if population == "adult" else "Pediatrics"
    path = redcap_root / "docs" / sub / "Acoustic Tasks" / fname
    text = path.read_text()
    # status is determined by which section (## Active / ## Retired) precedes a table
    sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
    for section in sections:
        status = "retired" if section.lower().lstrip().startswith(("retired", "\U0001f5c4")) or \
            "retired acoustic tasks" in section.lower()[:40] else "current"
        for header, rows in parse_gfm_tables(section):
            task_col = None
            for h in header:
                if h.strip().lower() == "task":
                    task_col = h
                    break
            if task_col is None:
                continue
            count_col = find_key(header, "recording")
            usedin_col = find_key(header, "used in") or find_key(header, "retired from") \
                or find_key(header, "age group")
            for row in rows:
                raw = row.get(task_col, "").strip()
                if not raw:
                    continue
                family, version = strip_version(raw)
                yield {
                    "population": population,
                    "status": status,
                    "family": family,
                    "version": version,
                    "recording_count": _to_int(row.get(count_col, "")),
                    "used_in": _split_used_in(row.get(usedin_col, "")),
                    "raw_task_cell": raw,
                }


def _to_int(text):
    m = re.search(r"\d+", str(text))
    return int(m.group(0)) if m else None


def _split_used_in(text):
    text = re.sub(r"\[|\]\([^)]*\)", "", str(text))  # drop md link targets
    parts = [p.strip() for p in re.split(r",|<br>", text) if p.strip()]
    return parts


# --------------------------------------------------------------------------- #
# Recording-index parsing (per-recording rows) + instruction harvesting
# --------------------------------------------------------------------------- #
def parse_recordings_index(redcap_root: Path, population: str):
    """Map family_slug -> ordered list of recording sub-names from Recordings.md.

    A recording name is "<Family>-<sub>"; the sub-name is what distinguishes the
    grouped sub-recordings (e.g. Conversation (6 plus)-favorite food).
    """
    fname = "Adult Recordings.md" if population == "adult" else "Pediatric Recordings.md"
    sub = "Adults" if population == "adult" else "Pediatrics"
    path = redcap_root / "docs" / sub / "Recordings" / fname
    out = OrderedDict()
    for header, rows in parse_gfm_tables(path.read_text()):
        name_col = find_key(header, "recording", "name")
        task_col = find_key(header, "acoustic", "task")
        if not name_col or not task_col:
            continue
        for row in rows:
            rec_name = clean_recording_name(row.get(name_col, ""))
            family = strip_version(row.get(task_col, "").strip())[0]
            if not rec_name or not family:
                continue
            fam_slug = slug(family)
            sub_name = rec_name[len(family):].lstrip("- ").strip() if rec_name.lower().startswith(
                family.lower()) else rec_name
            out.setdefault(fam_slug, []).append({"recording_name": rec_name, "sub_name": sub_name})
    return out


def build_flat_slug_index(flat):
    """Map slug(key) -> resolved instructions (following alias_of)."""
    index = {}
    for key, entry in flat.items():
        target = entry
        seen = set()
        while isinstance(target, dict) and "alias_of" in target:
            nxt = target["alias_of"]
            if nxt in seen or nxt not in flat:
                break
            seen.add(nxt)
            target = flat[nxt]
        index[slug(key)] = (target.get("instructions", "") if isinstance(target, dict) else "")
    return index


def harvest_recording_instructions(flat_slug_index, sub_name, family_slug):
    """Find instructions for a sub-recording by trying its slug, a curated map,
    and the conversation-book special case. Returns '' if not found."""
    candidates = []
    s = slug(sub_name) if sub_name else ""
    if s:
        candidates.append(s)
        if s in SUBTASK_TO_FLAT:
            candidates.append(slug(SUBTASK_TO_FLAT[s]))
    if family_slug in ("conversation-2-to-4", "conversation-4-to-6"):
        candidates.append(f"{family_slug}-book")
    for cand in candidates:
        if flat_slug_index.get(cand):
            return flat_slug_index[cand]
    return ""


def build_aliases(population, family, version):
    fam_slug = slug(family)
    aliases = {fam_slug, slug(f"{family} {version}") if version else fam_slug}
    if version == "v2":
        aliases.add(f"{fam_slug}-(v2)")
        aliases.add(f"{fam_slug}(v2)")
    return sorted(a for a in aliases if a)


# --------------------------------------------------------------------------- #
# Registry assembly
# --------------------------------------------------------------------------- #
def harvest_task_instructions(flat, family_slug, version):
    """Family-level instructions: the best flat entry whose key matches the family
    (following alias_of, preferring the matching version)."""
    bridge = FAMILY_TO_FLAT.get(family_slug)
    if bridge and bridge in flat and flat[bridge].get("instructions"):
        return flat[bridge]["instructions"]
    best, best_score = "", -1
    for key, entry in flat.items():
        target, seen = entry, set()
        while isinstance(target, dict) and "alias_of" in target:
            nxt = target["alias_of"]
            if nxt in seen or nxt not in flat:
                break
            seen.add(nxt)
            target = flat[nxt]
        instr = target.get("instructions", "") if isinstance(target, dict) else ""
        if not instr:
            continue
        kslug = slug(re.sub(r"\(v2\)", "", key))
        if kslug.startswith(family_slug) or family_slug.startswith(kslug):
            score = 2 if ((version == "v2") == ("(v2)" in key)) else 1
            if score > best_score:
                best, best_score = instr, score
    return best


def build_registry(redcap_root: Path, git_tag: str):
    flat = json.loads(FLAT_DESCRIPTIONS.read_text(), object_pairs_hook=OrderedDict)
    flat_slug_index = build_flat_slug_index(flat)
    tasks = OrderedDict()
    alias_index = OrderedDict()
    report = {"tasks": 0, "recordings": 0, "no_instructions": [], "unknown_policy": []}

    for population in ("adult", "pediatric"):
        recs_by_family = parse_recordings_index(redcap_root, population)
        for t in parse_task_index(redcap_root, population):
            fam_slug = slug(t["family"])
            task_id = f"{population}.{fam_slug}" + (f".{t['version']}" if t["version"] else "")
            speech_type, prompt_ref = POLICY.get(fam_slug, DEFAULT_POLICY)
            if fam_slug not in POLICY:
                report["unknown_policy"].append(fam_slug)

            task_instructions = harvest_task_instructions(flat, fam_slug, t["version"])

            # Genuinely grouped tasks (distinct sub-recordings, e.g. Conversation
            # (6 plus), Generative Naming) get nested recording entries, each with
            # its OWN instructions. Uniform-repeat tasks (loudness x2, DDK syllables)
            # keep only the task-level instruction. A task is "grouped" here when >=2
            # of its sub-recordings resolve to non-empty, distinct instructions.
            recordings = []
            rows = recs_by_family.get(fam_slug, [])
            if fam_slug not in BANKED_FAMILIES and len(rows) > 1:
                candidates = []
                for rec in rows:
                    instr = harvest_recording_instructions(flat_slug_index, rec["sub_name"], fam_slug)
                    candidates.append(OrderedDict([
                        ("recording_id", slug(rec["recording_name"])),
                        ("canonical_name", rec["recording_name"]),
                        ("sub_name", rec["sub_name"]),
                        ("instructions", instr),
                    ]))
                resolved = [c for c in candidates if c["instructions"]]
                if len({c["instructions"] for c in resolved}) >= 2:
                    recordings = candidates
                    report["recordings"] += len(candidates)

            if not task_instructions and not any(r["instructions"] for r in recordings):
                report["no_instructions"].append(task_id)

            aliases = build_aliases(population, t["family"], t["version"])
            entry = OrderedDict([
                ("task_id", task_id),
                ("canonical_name", t["raw_task_cell"]),
                ("population", population),
                ("status", t["status"]),
                ("family", t["family"]),
                ("version", t["version"]),
                ("age_or_cohort", t["used_in"]),
                ("recording_count", t["recording_count"]),
                ("instructions", task_instructions),
                ("speech_type", speech_type),
                ("prompt_ref", prompt_ref),
                ("recordings", recordings),
                ("aliases", aliases),
            ])
            tasks[task_id] = entry
            for a in aliases:
                alias_index.setdefault(a, task_id)
            report["tasks"] += 1

    registry = OrderedDict([
        ("schema_version", "1"),
        ("source", OrderedDict([("repo", "bridge2ai-redcap"), ("git_tag", git_tag)])),
        ("tasks", tasks),
        ("alias_index", alias_index),
    ])
    return registry, report


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--redcap-root", required=True, type=Path)
    ap.add_argument("--git-tag", default="unknown")
    ap.add_argument("--check", action="store_true", help="fail if committed output would change")
    args = ap.parse_args(argv)

    root = args.redcap_root
    if not root.exists():
        ap.error(f"redcap root not found: {root}")

    outputs = {
        REGISTRY_DIR / "harvard_sentences_bank.json": extract_harvard_bank(root),
        REGISTRY_DIR / "productive_vocabulary_bank.json": extract_productive_vocabulary_bank(root),
        REGISTRY_DIR / "random_item_categories.json": extract_random_categories(root),
        REGISTRY_DIR / "cape_v_sentences_bank.json": extract_cape_v_bank(),
    }
    registry, report = build_registry(root, args.git_tag)
    outputs[REGISTRY_DIR / "registry.json"] = registry

    if args.check:
        drift = []
        for path, obj in outputs.items():
            new = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
            old = path.read_text() if path.exists() else None
            if old != new:
                drift.append(path.name)
        if drift:
            print(f"DRIFT: committed registry differs: {drift}", file=sys.stderr)
            return 1
        print("registry up to date")
        return 0

    for path, obj in outputs.items():
        write_json(path, obj)

    hb = outputs[REGISTRY_DIR / "harvard_sentences_bank.json"]["lists"]
    print(f"wrote {len(outputs)} files to {REGISTRY_DIR}")
    print(f"  harvard lists: {len(hb)} (sentences: {sum(len(v) for v in hb.values())})")
    print(f"  tasks: {report['tasks']}")
    if report["unknown_policy"]:
        print(f"  families with DEFAULT policy: {sorted(set(report['unknown_policy']))}")
    if report["no_instructions"]:
        print(f"  tasks with NO instructions harvested: {report['no_instructions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

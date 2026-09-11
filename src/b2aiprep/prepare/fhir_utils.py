"""Utility functions for converting participant data (usually nested dictionaries)
to FHIR format.
"""

import json
import re
import typing as t
from collections import OrderedDict
from functools import lru_cache
from importlib.resources import files
from urllib.parse import quote
import logging

from b2aiprep.prepare.utils import canonical_task_entity
_logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _load_stimulus_bank(bank_name: str) -> str:
    """Load a stimulus bank JSON (as text, for hashable caching) from resources.

    Looks in prepare/resources/task_registry/ first (generated banks), then
    prepare/resources/ (the phase-1 repeating_words_bank). Returns the raw JSON
    text; callers parse it (kept as text so the lru_cache value stays hashable).
    """
    base = files("b2aiprep.prepare.resources")
    for parent in ("task_registry", ""):
        resource = base.joinpath(parent, f"{bank_name}.json") if parent \
            else base.joinpath(f"{bank_name}.json")
        try:
            return resource.read_text()
        except (FileNotFoundError, OSError):
            continue
    raise FileNotFoundError(f"stimulus bank not found: {bank_name}")


@lru_cache(maxsize=None)
def _bank(bank_name: str) -> dict:
    # Parsed once per bank and cached: the per-recording resolver hits the same
    # banks thousands of times in a large build. Callers read the dict (.get) and
    # must not mutate the shared instance.
    return json.loads(_load_stimulus_bank(bank_name))


@lru_cache(maxsize=None)
def _bank_name_for_language(base: str, language: str) -> str:
    """Return the language-specific bank name (`<base>_<lang>`, e.g.
    harvard_sentences_bank_es_419) when that bank exists, else the base
    (English) bank. So a language with no bank transparently falls back."""
    if not language or language == "en":
        return base
    candidate = f"{base}_{language.replace('-', '_').lower()}"
    try:
        _load_stimulus_bank(candidate)
        return candidate
    except FileNotFoundError:
        return base


def _resolve_prompt_ref(task_name: str, prompt_ref: dict, language: str = "en") -> str:
    """Resolve a task name to its specific stimulus via a stimulus bank.

    Dispatches on prompt_ref['select']:
    - index-or-word (default; repeating_words): trailing token -> word by 1-based
      index (e.g. repeat-words-24) or by literal word (e.g. repeating-words-slice).
    - index (reading_passage / repeating_sentences): trailing integer ->
      sentences[index-1].
    - list-index (harvard): '...-list-<L>-<N>' -> lists[L][N-1].
    - version-index (cape-v): version from '(v2)' + trailing <N> -> lists[version][N].
    Returns "" when it can't be resolved. `task_name` is the ORIGINAL name (parens
    intact) so the '(v2)' / 'list-L-N' patterns still match.
    """
    bank = _bank(_bank_name_for_language(prompt_ref["bank"], language))
    select = prompt_ref.get("select", "index-or-word")

    if select == "list-index":
        m = re.search(r"list-(\d+)-(\d+)$", task_name)
        if not m:
            return ""
        lst, idx = m.group(1), int(m.group(2))
        items = bank.get("lists", {}).get(lst, [])
        return items[idx - 1] if 1 <= idx <= len(items) else ""

    if select == "version-index":
        # The version token appears in either position in the data
        # ("cape-v-sentences-(v2)-4" and "cape-v-sentences-4-(v2)"); strip it, then
        # the remaining number is the sentence index.
        version = "v2" if "(v2)" in task_name else "v1"
        core = re.sub(r"\(v\d+\)", "", task_name)
        m = re.search(r"(\d+)", core)
        if not m:
            return ""
        lists = bank.get("lists", {})
        sub = lists.get(version)
        # A language bank may carry only the current-protocol version; a v1-labelled
        # non-English recording still read that (single) Spanish set, so fall back
        # to the only version present. English keeps both, so this never triggers.
        if not sub and lists:
            sub = next(iter(lists.values()))
        return (sub or {}).get(m.group(1), "")

    if select == "index":
        items = bank.get("sentences", bank.get("words", []))
        token = re.split(r"[-_]", task_name)[-1]
        if token.isdigit() and 1 <= int(token) <= len(items):
            return items[int(token) - 1]
        return ""

    # default: index-or-word over a flat "words" list
    words = bank.get("words", [])
    token = re.split(r"[-_]", task_name)[-1]
    if token.isdigit():
        idx = int(token)
        return words[idx - 1] if 1 <= idx <= len(words) else ""
    for word in words:
        if word.lower() == token.lower():
            return word
    return ""


# Provisional speech-type classification by task-name family (phase 1). It tells
# a downstream user how the produced speech relates to a reference:
#   read        - verbatim reading of provided text (stimulus_text is WER ground truth)
#   recall      - retelling from memory of a reference (stimulus_text = reference, not verbatim)
#   elicited    - spontaneous/prompted, no target text (default)
#   non-lexical - vocalizations without lexical content (DDK, vowels, sounds, cough...)
# An entry may override this with an explicit "speech_type". Phase 2's registry
# generator will assign speech_type per family authoritatively.
_SPEECH_TYPE_PREFIXES = (
    ("harvard-sentences", "read"),
    ("cape-v-sentences", "read"),
    ("repeat-words", "read"),
    ("repeating-words", "read"),
    ("sentence", "read"),
    ("caterpillar-passage", "read"),
    ("passage", "read"),
    ("rainbow", "read"),
    ("story-recall", "recall"),
    ("cinderella-story", "recall"),
    ("diadochokinesis", "non-lexical"),
    ("prolonged-vowel", "non-lexical"),
    ("glides", "non-lexical"),
    ("high-to-low", "non-lexical"),
    ("loudness", "non-lexical"),
    ("maximum-phonation-time", "non-lexical"),
    ("respiration-and-cough", "non-lexical"),
    ("voluntary-cough", "non-lexical"),
    ("breath-sounds", "non-lexical"),
    ("noisy-sounds", "non-lexical"),
    ("silly-sounds", "non-lexical"),
    ("long-sounds", "non-lexical"),
)


def _classify_speech_type(key: str) -> str:
    """Infer speech_type from a resolved description key (see _SPEECH_TYPE_PREFIXES)."""
    k = key.lower()
    for prefix, speech_type in _SPEECH_TYPE_PREFIXES:
        if k.startswith(prefix):
            return speech_type
    return "elicited"


def _is_present(value) -> bool:
    """True when a questionnaire cell holds a real value (not None/NaN/empty)."""
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in ("", "nan", "none")


def _trailing_index(task_name: str):
    """The trailing integer of a task name (e.g. productive-vocabulary-3 -> 3), else None."""
    token = re.split(r"[-_]", task_name)[-1]
    return int(token) if token.isdigit() else None


def _stimulus_text_from_questionnaire(best_task, task_name, join_id, questionnaire_lookup):
    """Resolve per-participant stimulus_text for tasks whose stimulus lives in a
    linked questionnaire, joined on the recording's acoustic-task id.

    Returns (stimulus_text, speech_type, instructions_suffix) or None when the task
    is not questionnaire-backed or no matching row is found. `questionnaire_lookup`
    is keyed by (instrument, acoustic_task_id).
    """
    if not questionnaire_lookup or not _is_present(join_id):
        return None
    key = best_task.lower()
    if key.startswith("productive-vocabulary"):
        row = questionnaire_lookup.get(("vocab", join_id))
        idx = _trailing_index(task_name)
        if row is None or idx is None:
            return None
        word = row.get(f"vocabulary_item_word_{idx}")
        return (str(word).strip() if _is_present(word) else "", "elicited", None)
    if key.startswith("random-item-generation"):
        row = questionnaire_lookup.get(("random", join_id))
        if row is None:
            return None
        category = row.get("random_item_generation_category")
        suffix = f"Category: {str(category).strip()}." if _is_present(category) else None
        return ("", "elicited", suffix)
    if key.startswith("word-color-stroop"):
        row = questionnaire_lookup.get(("stroop", join_id))
        if row is None:
            return None
        colors = [row.get(f"stroop_item_color_{i}") for i in range(1, 16)]
        colors = [str(c).strip() for c in colors if _is_present(c)]
        # participant names the ink colors in order -> a known target sequence
        return (" ".join(colors), "read", None)
    return None


# --------------------------------------------------------------------------- #
# Registry-first resolution (phase 2c). The vendored task_registry/registry.json
# mirrors the bridge2ai-redcap hierarchy and is authoritative for instructions,
# speech_type, and prompt_ref. Resolution is registry-first with a three-tier
# fallback (registry -> flat dict -> substring): anything the registry cannot
# produce (e.g. static passage text it does not carry) falls back to the flat
# audio_task_descriptions.json so currently-populated sidecars do not regress.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _load_registry() -> dict:
    """Load the vendored task registry, or {} when it is not packaged."""
    try:
        resource = files("b2aiprep.prepare.resources").joinpath(
            "task_registry", "registry.json"
        )
        return json.loads(resource.read_text())
    except (FileNotFoundError, OSError):
        return {}


def _norm(name: str) -> str:
    """Normalize a task/recording name for matching: lowercase, drop parentheses,
    collapse runs of '-'. So 'Conversation-(6-plus)-favorite-food' and the
    registry recording_id 'conversation-6-plus-favorite-food' compare equal."""
    return re.sub(r"-+", "-", re.sub(r"[()]", "", name.lower())).strip("-")


@lru_cache(maxsize=None)
def _alias_items() -> tuple:
    """(normalized_alias, alias_len, task_id) tuples, longest alias first, so a
    single pass yields the longest-substring match deterministically."""
    reg = _load_registry()
    items = [(_norm(a), tid) for a, tid in reg.get("alias_index", {}).items()]
    items.sort(key=lambda it: len(it[0]), reverse=True)
    return tuple((na, len(na), tid) for na, tid in items)


@lru_cache(maxsize=None)
def _family_version_index() -> dict:
    """(population, family_slug) -> {version: task_id}. version is 'v1'/'v2'/...
    from the task_id suffix, or '' when unversioned. Lets resolution pick the
    version the recording name indicates rather than whichever the alias_index
    happened to keep first."""
    reg = _load_registry()
    index: dict = {}
    for tid, task in reg.get("tasks", {}).items():
        m = re.search(r"\.(v\d+)$", tid)
        version = m.group(1) if m else ""
        key = (task.get("population"), _norm(task.get("family", "")))
        index.setdefault(key, {})[version] = tid
    return index


def _select_version(reg, task, task_name):
    """When a family has multiple versions, pick the one the NAME indicates:
    '(v2)' -> v2, else the unversioned/v1 task. Fixes the alias_index collision
    where e.g. bare 'maximum-phonation-time' resolved to v2 (its recordings are
    fewer) rather than v1."""
    key = (task.get("population"), _norm(task.get("family", "")))
    versions = _family_version_index().get(key)
    if not versions or len(versions) < 2:
        return task
    # Detect the version token in every form it occurs: "(v2)" (real task names),
    # a bare "v2" token (paren-free recording_ids), and the glued "...timev2" form
    # that _norm() produces when parens are stripped without a separator. Require a
    # digit run not followed by another letter so "cape-v-sentences" (v then '-')
    # and family names are never mistaken for a version. Generalized to any vN.
    m = re.search(r"v(\d+)(?![a-z0-9])", task_name.lower())
    want = f"v{m.group(1)}" if m else None
    if want and want in versions:
        return reg["tasks"][versions[want]]
    if want is None:
        for fallback in ("", "v1"):  # version-less name -> unversioned, else v1
            if fallback in versions:
                return reg["tasks"][versions[fallback]]
    return task


@lru_cache(maxsize=None)
def _family_population_index() -> dict:
    """(population, family_slug) -> task_id, for routing a recording to the task
    in its own population when a family exists in more than one (only
    picture-description today)."""
    reg = _load_registry()
    index: dict = {}
    for tid, task in reg.get("tasks", {}).items():
        index.setdefault((task.get("population"), _norm(task.get("family", ""))), tid)
    return index


def _population_from_cohort(cohort):
    """Map an acoustic_task_cohort value to a population. Pediatric cohorts are
    'pediatric' or 'age_*'; every other (generic/neurology/voice/respiratory/mood)
    is adult. Returns None when unknown (no cohort) -> no population preference."""
    if not cohort:
        return None
    c = str(cohort).strip().lower()
    if c == "pediatric" or c.startswith("age_") or c.startswith("age-"):
        return "pediatric"
    return "adult"


# Administration-language codes. The RedCap `selected_language` records the
# language a session was run in ("English", "Spanish", ...); map it to a BCP-47
# code that matches the bridge2ai-redcap translation layer (es-419 = the project's
# Latin-American Spanish). Unknown/blank -> the release default, English.
DEFAULT_LANGUAGE = "en"
# The RedCap `selected_language` is a radio with exactly three choices
# (data dictionary: "1, English | 2, French | 3, Spanish"); `selected_language_2`
# carries the same three as BCP-47 codes ("en-US, English | fr-CA, French |
# es-419, Spanish"). Map every form an export can carry -- text labels, the
# integer codes of a coded export, and the BCP-47 codes -- so a coded export
# never silently collapses Spanish/French to English.
_LANGUAGE_CODES = {
    # text labels
    "english": "en", "french": "fr-CA", "spanish": "es-419",
    "espanol": "es-419", "español": "es-419", "français": "fr-CA", "francais": "fr-CA",
    # selected_language integer codes (1/2/3)
    "1": "en", "2": "fr-CA", "3": "es-419",
    # BCP-47 (selected_language_2 codes, and normalized outputs)
    "en": "en", "en-us": "en", "fr": "fr-CA", "fr-ca": "fr-CA",
    "es": "es-419", "es-419": "es-419",
}


def _language_from_selected(value) -> str:
    """Map a RedCap `selected_language` value to a BCP-47 code (default 'en').

    A value that is present but unrecognized (a new language, or a coded/integer
    export instead of labels) is NOT silently treated as English -- it is logged,
    because defaulting a non-English session to 'en' pairs English stimulus with
    non-English audio, the exact failure this feature prevents."""
    if value is None:
        return DEFAULT_LANGUAGE
    key = str(value).strip().lower()
    if key in ("", "nan", "none"):
        return DEFAULT_LANGUAGE
    code = _LANGUAGE_CODES.get(key)
    if code is None:
        _logger.warning(
            "unrecognized selected_language=%r; defaulting to %r. If the export is "
            "coded (integers) or a new language was added, extend _LANGUAGE_CODES.",
            value, DEFAULT_LANGUAGE,
        )
        return DEFAULT_LANGUAGE
    return code


def _select_population(reg, task, population):
    """When a family exists in more than one population, route to the task in the
    recording's population (e.g. peds vs adult picture-description)."""
    if not population or task.get("population") == population:
        return task
    tid = _family_population_index().get((population, _norm(task.get("family", ""))))
    return reg["tasks"][tid] if tid else task


def _resolve_task_registry(task_name: str, population=None):
    """Resolve a (granular) task name to (task_entry, recording_entry|None) using
    the registry alias_index (exact, else longest normalized-substring), corrected
    to the version the name indicates, then the nested recording whose
    recording_id equals the normalized task name. Returns None when the registry
    is absent or nothing matches."""
    reg = _load_registry()
    if not reg:
        return None
    # Resolve through the curated recording-name aliases first, so a variant spelling finds
    # the same task as its canonical form. Only aliased names are affected: for every other
    # name canonical_task_entity() differs from _norm() by punctuation that _norm() already
    # strips. Measured over all 949 task names in the v4 exports, exactly one resolution
    # changes -- the bare "High to Low", the Glides pair's second half, which resolved to
    # nothing and now resolves to adult.glides.
    n = _norm(canonical_task_entity(task_name))
    best_tid, best_len = None, -1
    for na, na_len, tid in _alias_items():
        if na == n:
            best_tid, best_len = tid, na_len
            break
        if na in n and na_len > best_len:
            best_tid, best_len = tid, na_len
    if best_tid is None:
        return None
    task = _select_version(reg, reg["tasks"][best_tid], task_name)
    task = _select_population(reg, task, population)
    rec = None
    for candidate in task.get("recordings", []):
        if _norm(candidate.get("recording_id", "")) == n:
            rec = candidate
            break
    return task, rec


@lru_cache(maxsize=None)
def _random_item_instructions() -> dict:
    """Curated per-language random-item instruction texts (general + category)."""
    try:
        return json.loads(_load_stimulus_bank("random_item_instructions"))
    except FileNotFoundError:
        return {}


def _random_item_instruction(category, language):
    """The random-item instruction for a recording, chosen by its category.

    A category outside {Numbers, Letters} appears only in Category_2 -> it is
    unambiguously the non-repeatable category variant, so use the category-variant
    instruction with the category named. Numbers/Letters appear in BOTH category
    lists, so the variant (repeatable vs not) is not determinable from the data --
    those keep the 'general' instruction, which describes both variants without
    asserting a repeatability we cannot confirm. Returns None if no resource."""
    data = _random_item_instructions()
    entry = data.get(language) or data.get("en")
    if not entry:
        return None
    cat = str(category).strip() if category else None
    label = entry.get("category_label", "Category")
    suffix = f"{label}: {cat}." if cat else ""
    if cat and cat.lower() not in ("numbers", "letters"):
        # unambiguously the non-repeatable category variant (Category_2 only)
        parts = [entry.get("category", ""), entry.get("procedural", ""), suffix]
    else:
        # Numbers/Letters (in both category lists) or no category: the general
        # instruction (describes both variants); still surface the drawn category.
        parts = [entry.get("general", ""), suffix]
    return " ".join(p for p in parts if p).strip() or None


def _registry_questionnaire(prompt_ref, task_name, join_id, questionnaire_lookup, language="en"):
    """Per-participant stimulus for a questionnaire-join prompt_ref. Returns
    (stimulus_text, speech_type, instructions_suffix, instructions_override) or
    None. instructions_override, when set, replaces the task instruction entirely
    (used for random-item, whose instruction depends on the drawn category)."""
    if not questionnaire_lookup or not _is_present(join_id):
        return None
    row = questionnaire_lookup.get((prompt_ref.get("instrument"), join_id))
    if row is None:
        return None
    select = prompt_ref.get("select")
    if select == "index-field":
        idx = _trailing_index(task_name)
        if idx is None:
            return None
        word = row.get(prompt_ref["field_template"].format(i=idx))
        return (str(word).strip() if _is_present(word) else "", "elicited", None, None)
    if select == "field":
        value = row.get(prompt_ref["field"])
        cat = str(value).strip() if _is_present(value) else None
        instr = _random_item_instruction(cat, language)
        # The variant-specific instruction embeds the category, so no extra suffix;
        # if there is no instruction resource, fall back to surfacing the category.
        suffix = None if instr else (f"Category: {cat}." if cat else None)
        return ("", "elicited", suffix, instr)
    if select == "color-list":
        n = prompt_ref.get("n", 15)
        colors = [row.get(prompt_ref["field_template"].format(i=i)) for i in range(1, n + 1)]
        colors = [str(c).strip() for c in colors if _is_present(c)]
        return (" ".join(colors), "read", None, None)
    return None


def _raw_github_url(prompt_ref, path, safe="/"):
    """Build a commit-pinned raw GitHub URL from a prompt_ref's asset_repo/asset_commit
    and an (already resolved) path, URL-encoding the path. `safe` keeps extra
    characters unescaped -- the default '/' preserves separators; '/{}' additionally
    preserves a literal '{n}' template. Returns None when repo/commit/path are
    missing. Single source for both the single-image and sequence URL forms."""
    repo, commit = prompt_ref.get("asset_repo"), prompt_ref.get("asset_commit")
    if not path or not repo or not commit:
        return None
    return "https://raw.githubusercontent.com/{repo}/{commit}/{path}".format(
        repo=repo, commit=commit, path=quote(path, safe=safe),
    )


def _asset_url(prompt_ref, task_name):
    """A commit-pinned raw GitHub URL for the recording's SINGLE, directly-resolvable
    image stimulus, or None. Resolves the path three ways:
    - asset_map: keyed by the recording's trailing token (e.g. picture-description
      'option1'/'option2');
    - asset_path with '{i}'/'{i:02d}': filled from the trailing index;
    - asset_path without a placeholder: a single fixed image.
    A '{n}'-templated path is a multi-image SEQUENCE, not a single image, so this
    returns None regardless of task type -- _asset_sequence_url owns those (prevents
    a '{n}' path from being percent-encoded into a broken single URL)."""
    path = None
    asset_map = prompt_ref.get("asset_map")
    if asset_map:
        path = asset_map.get(re.split(r"[-_]", task_name)[-1])
    if path is None:
        p = prompt_ref.get("asset_path")
        if p and "{i" in p:
            idx = _trailing_index(task_name)
            path = p.format(i=idx) if idx is not None else None
        else:
            path = p
    if path and "{n" in path:
        return None
    return _raw_github_url(prompt_ref, path)


def _asset_sequence_url(prompt_ref):
    """A single SCALAR template URL for an ordered multi-image stimulus (e.g. Story
    Recall v2's 10 wordless panels the participant views before retelling). Kept
    scalar -- not a list -- so the sidecar metadata stays flat/parquet-friendly: the
    URL carries a literal '{n}' placeholder and the companion sidecar field
    'stimulus_asset_n_images' gives N, so a consumer substitutes n = 1..N to get the
    panel URLs. The panels are wordless, so this is language-independent (the same
    sequence for every language; only stimulus_text differs). Returns None unless
    prompt_ref carries a '{n}'-templated asset_path and a positive asset_count.
    Distinct from _asset_url's per-recording '{i}' single, directly-resolvable image
    (which has no '{n}' and no n-images count)."""
    p = prompt_ref.get("asset_path")
    n = prompt_ref.get("asset_count")
    if not p or "{n" not in p or not isinstance(n, int) or n < 1:
        return None
    # keep '/' separators and the literal '{n}' placeholder unescaped
    return _raw_github_url(prompt_ref, p, safe="/{}")


def _registry_numbering_status(task, task_name):
    """Validate the recording index in `task_name` against the task's known
    numbering, so impossible instances surface (e.g. a 5th maximum-phonation-time,
    a repeating-sentences-9). Returns 'ok', 'out-of-range', or 'unknown' (can't
    tell). The registry is the authority on how many recordings/indices exist:
    banked tasks are bounded by their bank, grouped tasks by their nested
    recordings, uniform tasks by recording_count."""
    prompt_ref = task.get("prompt_ref") or {}
    ptype = prompt_ref.get("type")
    select = prompt_ref.get("select")

    if ptype == "stimulus-bank":
        bank = _bank(prompt_ref["bank"])
        if select == "list-index":
            m = re.search(r"list-(\d+)-(\d+)$", task_name)
            if not m:
                return "out-of-range"
            items = bank.get("lists", {}).get(m.group(1))
            return "ok" if items and 1 <= int(m.group(2)) <= len(items) else "out-of-range"
        if select == "version-index":
            version = "v2" if "(v2)" in task_name else "v1"
            m = re.search(r"(\d+)", re.sub(r"\(v\d+\)", "", task_name))
            if not m:
                return "out-of-range"
            return "ok" if m.group(1) in bank.get("lists", {}).get(version, {}) else "out-of-range"
        if select == "index":
            items = bank.get("sentences", bank.get("words", []))
            idx = _trailing_index(task_name)
            return "ok" if idx is not None and 1 <= idx <= len(items) else "out-of-range"
        if select == "index-or-word":
            words = bank.get("words", [])
            token = re.split(r"[-_]", task_name)[-1]
            if token.isdigit():
                return "ok" if 1 <= int(token) <= len(words) else "out-of-range"
            return "ok" if any(w.lower() == token.lower() for w in words) else "out-of-range"

    recordings = task.get("recordings") or []
    if recordings:
        n = _norm(task_name)
        if any(_norm(r.get("recording_id", "")) == n for r in recordings):
            return "ok"
        idx = _trailing_index(task_name)
        if idx is not None and idx > len(recordings):
            return "out-of-range"
        return "ok"  # matched the family but no specific recording (task-level)

    count = task.get("recording_count")
    idx = _trailing_index(task_name)
    if idx is not None and isinstance(count, int) and count > 0 and idx > count:
        return "out-of-range"
    return "ok"


@lru_cache(maxsize=None)
def _language_resource(name: str, language: str) -> str:
    """A per-language resource bank (`<name>_<lang>`) as text, or '{}' when absent."""
    try:
        return _load_stimulus_bank(f"{name}_{language.replace('-', '_').lower()}")
    except FileNotFoundError:
        return "{}"


def _family_slug(family) -> str:
    """Registry family name ('Cape V Sentences') -> bank key slug ('cape-v-sentences')."""
    return str(family or "").strip().lower().replace(" ", "-")


def _static_stimulus(family, language):
    """The static-inline stimulus entry (passage/recall reference) for a task
    family in a language, or None. Keyed by family so any version of the family a
    non-English session recorded picks up the current-protocol reference."""
    if not language or language == "en" or not family:
        return None
    return json.loads(_language_resource("static_stimulus", language)).get(_family_slug(family))


def _language_instructions(family, language):
    """The Spanish/other-language instruction for a task family, or None."""
    if not language or language == "en" or not family:
        return None
    return json.loads(_language_resource("task_instructions", language)).get(_family_slug(family))


def _free_speech_cue(task_name, language):
    """Per-recording free-speech cue in a language, for the NUMBERED current (v2)
    task only. The voice variant (unnumbered "free-speech") and v1 have no es-419
    source, so they keep the English cue -- keyed on the '(v2)' marker + trailing
    index so those are never given the v2 questions."""
    if not language or language == "en" or "(v2)" not in task_name:
        return None
    idx = _trailing_index(task_name)
    if idx is None:
        return None
    return json.loads(_language_resource("free_speech_bank", language)).get(str(idx))


def _registry_bids_fields(task, rec, task_name, join_id, questionnaire_lookup, language="en"):
    """Assemble sidecar fields from a registry match. `stimulus_text` is None when
    the registry cannot produce it (static-inline / image tasks carry no text) so
    the caller can fall back to the flat file. Returns a dict."""
    # Instructions are authoritative only when curated (per-recording, or a
    # curated task-level instruction such as reading-passage). A coarse harvested
    # task-level instruction is NOT authoritative: the flat file's per-key
    # instruction (e.g. diadochokinesis's per-syllable text, loudness v1/v2) is
    # more specific and must win. The caller consults `instructions_authoritative`.
    # A language-specific instruction (harvested from the es-419 task description)
    # wins for non-English sessions -- the participant received it in that language.
    lang_instr = _language_instructions(task.get("family"), language)
    if lang_instr:
        instructions = lang_instr
        instructions_authoritative = True
    elif rec and rec.get("instructions"):
        instructions = rec["instructions"]
        instructions_authoritative = True
    else:
        instructions = task.get("instructions", "") or ""
        instructions_authoritative = task.get("instructions_source") == "curated"
    speech_type = task.get("speech_type") or "elicited"
    prompt_ref = task.get("prompt_ref") or {}
    ptype = prompt_ref.get("type")
    stimulus_text = None
    stimulus_source = None
    stimulus_asset = None
    stimulus_asset_n_images = None
    instructions_suffix = None

    if ptype == "stimulus-bank":
        stimulus_text = _resolve_prompt_ref(task_name, prompt_ref, language)
        # A read/recall bank with no <bank>_<lang> variant falls back to the English
        # bank; flag it so a non-English read task never silently ships an English
        # reference tagged as another language (the mismatch WER would score against).
        base_bank = prompt_ref.get("bank")
        if (language and language != "en" and speech_type in ("read", "recall")
                and _bank_name_for_language(base_bank, language) == base_bank):
            _logger.warning(
                "no %s stimulus bank for %r; %r (%s) emits the English reference "
                "tagged language=%s -- add a %s_%s bank or expect a WER mismatch",
                language, base_bank, task_name, speech_type, language,
                base_bank, language.replace("-", "_"),
            )
        # reading/repeating-sentences text was transcribed from image stimuli (and
        # carries a pinned URL to that image); the lexical banks
        # (harvard/cape-v/repeating-words) are redcap doc text with no asset.
        stimulus_source = "transcribed" if prompt_ref.get("bank") in (
            "reading_passage_bank", "repeating_sentences_bank"
        ) else "doc-text"
        stimulus_asset = _asset_url(prompt_ref, task_name)
    elif ptype == "questionnaire-join":
        joined = _registry_questionnaire(prompt_ref, task_name, join_id, questionnaire_lookup, language)
        if joined is not None:
            stimulus_text, speech_type, instructions_suffix, instr_override = joined
            stimulus_source = "questionnaire"
            # random-item's instruction depends on the drawn category (and language),
            # so it fully overrides the task-level instruction when resolved.
            if instr_override:
                instructions = instr_override
                instructions_authoritative = True
        else:
            stimulus_text = ""
            stimulus_source = "questionnaire"
    elif ptype == "image":
        # The stimulus is a shown image; pin it to a commit-stable URL when the
        # prompt_ref carries a per-index asset path.
        stimulus_text = ""
        stimulus_source = "image"
        stimulus_asset = _asset_url(prompt_ref, task_name)
        # Some image cards also print the target on the card itself -- the word to
        # read (Identifying Pictures) or the sound to make (Noisy/Silly/Long
        # Sounds). That target is in neither the task name nor the instructions, so
        # it is transcribed once into an index-keyed text_bank (words[i-1] for the
        # trailing index). When present, emit it as the stimulus_text and keep the
        # image asset; provenance is 'transcribed' (read off the image). speech_type
        # stays whatever the task declares (read for Identifying Pictures, so it is
        # WER-scorable; non-lexical for the sound cards).
        text_bank = prompt_ref.get("text_bank")
        if text_bank:
            idx = _trailing_index(task_name)
            items = _bank(text_bank).get("words", [])
            if idx is not None and 1 <= idx <= len(items):
                stimulus_text = items[idx - 1]
                stimulus_source = "transcribed"
    # static-inline: stimulus_text stays None -> flat-file fallback (so read/recall
    # tasks pick up their passage text). EXCEPT non-lexical tasks, which have no
    # lexical reference: force "" so we don't leak the flat file's instruction
    # (stored there under "prompts") into stimulus_text (respiration, loudness,
    # DDK, prolonged-vowel, glides, cough, breath-sounds, ...).
    if stimulus_text is None and speech_type == "non-lexical":
        stimulus_text = ""

    # Non-English static-inline read/recall reference text (passages, story recall)
    # lives in a language-specific static bank, not the English flat file. Use it
    # so a Spanish session's read/recall recording carries its Spanish reference
    # rather than inheriting the English passage.
    if stimulus_text is None:
        # Free Speech (numbered v2) carries a per-recording Spanish cue; the voice
        # (unnumbered) and v1 variants have no es-419 source and keep the English
        # cue. Scoped to the Free Speech family so no other "(v2)-N" static-inline
        # task is ever handed the free-speech question.
        cue = (_free_speech_cue(task_name, language)
               if _family_slug(task.get("family")) == "free-speech" else None)
        if cue:
            stimulus_text = cue
        else:
            static = _static_stimulus(task.get("family"), language)
            if static and static.get("stimulus_text"):
                stimulus_text = static["stimulus_text"]
                stimulus_source = "doc-text"

    # Symmetric with the stimulus-bank path: a non-English read/recall task with no
    # language-specific reference will fall back to the English flat-file text in the
    # caller -- flag that mismatch rather than shipping it silently.
    if stimulus_text is None and language and language != "en" and speech_type in ("read", "recall"):
        _logger.warning(
            "no %s reference for read/recall task %r; the English text will be emitted "
            "tagged language=%s -- add a language static/bank entry or expect a WER mismatch",
            language, task_name, language,
        )

    # An ordered multi-image sequence stimulus (Story Recall v2's wordless panels)
    # is recorded as ONE scalar template URL plus an n-images count -- resolved after
    # the language-specific text and independent of language (the panels are
    # wordless). A single directly-resolvable image leaves n_images None (the field
    # is then omitted from the sidecar, which itself documents "resolvable as-is").
    if stimulus_asset is None:
        seq = _asset_sequence_url(prompt_ref)
        if seq:
            stimulus_asset = seq
            stimulus_asset_n_images = prompt_ref.get("asset_count")

    return {
        "instructions": instructions,
        "instructions_authoritative": instructions_authoritative,
        "speech_type": speech_type,
        "stimulus_text": stimulus_text,
        "stimulus_source": stimulus_source,
        "stimulus_asset": stimulus_asset,
        "stimulus_asset_n_images": stimulus_asset_n_images,
        "instructions_suffix": instructions_suffix,
    }


def _flat_bids_fields(task_name_lower, audio_task_descriptions, join_id, questionnaire_lookup, language="en"):
    """The flat audio_task_descriptions.json resolution (exact -> longest
    substring, alias_of hop, prompt_ref/static stimulus, questionnaire join).
    Returns a dict of fields, or None when no key matches."""
    best_task = None
    for task in audio_task_descriptions:
        task_lower = task.lower()
        if task_lower == task_name_lower:
            best_task = task
            break
        if task_lower in task_name_lower and (best_task is None or len(task) > len(best_task)):
            best_task = task
    if best_task is None:
        return None

    description = audio_task_descriptions[best_task]
    seen_aliases = set()
    while isinstance(description, dict) and "alias_of" in description:
        target = description["alias_of"]
        if target in seen_aliases or target not in audio_task_descriptions:
            break
        seen_aliases.add(target)
        description = audio_task_descriptions[target]
        # Classify (and join) on the resolved canonical key, not the alias name,
        # so an alias whose name doesn't share its target's prefix (e.g.
        # "cinderella-retell" -> a recall-family target) is not misclassified.
        best_task = target

    fields = {
        "instructions": description.get("instructions", ""),
        "speech_type": description.get("speech_type") or _classify_speech_type(best_task),
        "stimulus_text": "",
        "instructions_suffix": None,
    }
    prompt_ref = description.get("prompt_ref")
    if prompt_ref:
        fields["stimulus_text"] = _resolve_prompt_ref(task_name_lower, prompt_ref, language)
    elif "stimulus_text" in description:
        fields["stimulus_text"] = description["stimulus_text"]
    else:
        static_prompts = description.get("prompts", [])
        fields["stimulus_text"] = " ".join(static_prompts)

    if questionnaire_lookup:
        joined = _stimulus_text_from_questionnaire(
            best_task, task_name_lower, join_id, questionnaire_lookup
        )
        if joined is not None:
            q_text, q_type, q_suffix = joined
            fields["stimulus_text"] = q_text
            fields["speech_type"] = q_type
            fields["instructions_suffix"] = q_suffix
    return fields


def extract_items(participant_json: dict, outline: list) -> t.List[dict]:
    """Iterates over questions specified in the outline and extracts them from the data JSON.

    Items are output as a list of dictionaries, following the FHIR QuestionnaireResponse format
    for a single item.

    NOTE: Due to issues in the conversion, this function is additionally doing data cleaning.
    This includes mapping binary responses to the "valueBinary" field, and parsing instances of
    the "nan" string as missing data.

    Parameters
    ----------
    participant_json : dict
        The JSON data for the participant.
    outline : list
        The list of questions to extract.

    Returns
    -------
    list
        The extracted items.
    """
    items = []
    for question in outline:
        answer_value = str(participant_json[question])
        if answer_value in ("Checked", "Unchecked"):
            answer = answer_value == "Checked"
        elif answer_value == "nan":
            answer = None
        else:
            answer = answer_value.replace("_", "-")
        item = {
            "metadata": question,
            "answer": answer,
        }
        items.append(item)
    return items


def is_invalid_response(participant: dict, outline) -> bool:
    """Determines whether a questionnaire contains only NaN/unchecked responses.
    This indicates that the questionnaire was likely unpopulated by the participant.

    Parameters
    ----------
    participant : dict
        The participant data.
    outline : list
        The questionnaire outline.

    Returns
    -------
    bool
        Whether the response is invalid.
    """
    generic_items = extract_items(participant, outline)
    answers = []
    for item in generic_items:
        if item["answer"] is None:
            continue
        
        if isinstance(item["answer"], str):
            answers.append(item["answer"].lower())
        elif isinstance(item["answer"], bool):
            if item["answer"]:
                answers.append("checked")
            else:
                answers.append("unchecked")
        else:
            answers.append(item["answer"])
            
    answers = set(answers)
    return len(answers.difference({"nan", "unchecked"})) == 0


def convert_response_to_bids_metadata( participant: dict,
    # repeat_instrument: RepeatInstrument
    questionnaire_name: str,
    mapping_name: str,
    columns: t.List[str],
    audio_task_descriptions: OrderedDict,
    questionnaire_lookup: t.Optional[dict] = None,
    population: t.Optional[str] = None,
    language: t.Optional[str] = None,
) -> dict:
    """Converts a participant's response to a metadata json file.

    Given a dictionary of individual data, the function:
    1. identifies the audio task, recording id, duration, and task instructions

    Parameters
    ----------
    participant : dict
        The participant data.
    questionnaire_name : str
        The name of the questionnaire.

    Returns
    -------
    dict containing the associated metadata

    Raises
    ------
    ValueError
        If the parsed JSON does not adhere to the QuestionnaireResponse data structure.
        Usually due to missing or invalid attributes.
    """
    participant_id = participant["record_id"]

    generic_items = extract_items(participant, columns)
    if is_invalid_response(participant, columns):
        generic_items = []

    # determine if it's an acoustic task or recording and grab task names
    linkid_to_find = None
    if mapping_name == "acoustictaskschema":
        linkid_to_find = "acoustic_task_id"
    elif mapping_name == "recordingschema":
        linkid_to_find = "recording_id"
    
    if not linkid_to_find:
        _logger.warning(f"File is missing acoustic_task_id and recording_id, skipping....")
        return {}
    
    metadata_file = {}
    task_name = ""
    lang = language or DEFAULT_LANGUAGE
    # Administration language of the session, recorded on every sidecar so a
    # consumer can tell (and filter) which recordings are non-English -- and so a
    # WER pipeline never scores non-English speech against an English reference.
    metadata_file["language"] = lang
    metadata_file["instructions"] = ""
    for item in generic_items:
        metadata_field = item.get("metadata")
        metadata_value = item.get("answer")
        if "session_id" in metadata_field:
            metadata_file["session_id"] = metadata_value
        elif metadata_field in ("acoustic_task_name", "recording_name") and metadata_field is not None:
            if isinstance(metadata_value, str):
                task_name = metadata_value.replace(" ", "-").lower()
            metadata_file[metadata_field] = task_name
        elif metadata_file is not None and metadata_field is not None:
            metadata_file[metadata_field] = metadata_value
    
    if "recording_name" in metadata_file:
        metadata_file["task_name"] = metadata_file["recording_name"]
        metadata_file.pop("recording_name")
    # Resolve the task metadata registry-first, with a flat-file fallback.
    #
    # Tier 1: the vendored task_registry/registry.json (mirrors bridge2ai-redcap)
    #   is authoritative for instructions (per-recording where grouped),
    #   speech_type, and the stimulus prompt_ref (bank / questionnaire-join).
    # Tier 2/3: the flat audio_task_descriptions.json (exact -> longest substring,
    #   alias_of hop) supplies anything the registry cannot -- notably the static
    #   passage/recall text the registry does not carry -- so currently-populated
    #   sidecars never regress.
    #
    # Numbered instances of a task (e.g. "picture-12") share a single key/entry;
    # matching by longest substring keeps the lookup independent of ordering, so
    # e.g. "harvard-sentences-list-1-10" resolves to its own list/index rather
    # than being shadowed by the shorter "harvard-sentences-list-1-1".
    if task_name:
        task_name_lower = task_name.lower()
        join_id = participant.get("recording_acoustic_task_id")
        flat = _flat_bids_fields(
            task_name_lower, audio_task_descriptions, join_id, questionnaire_lookup, lang
        )
        reg_match = _resolve_task_registry(task_name_lower, population)

        resolved = None
        if reg_match is not None:
            task, rec = reg_match
            reg = _registry_bids_fields(
                task, rec, task_name_lower, join_id, questionnaire_lookup, lang
            )
            # Registry instructions win only when authoritative (curated);
            # otherwise the flat file's per-key instruction is more specific.
            if reg["instructions_authoritative"] and reg["instructions"]:
                instructions = reg["instructions"]
            elif flat and flat["instructions"]:
                instructions = flat["instructions"]
            else:
                instructions = reg["instructions"]
            # registry stimulus_text is None for static-inline/image tasks it does
            # not carry -> fall back to the flat file's text.
            stim_text = (
                reg["stimulus_text"]
                if reg["stimulus_text"] is not None
                else (flat["stimulus_text"] if flat else "")
            )
            stim_source = reg["stimulus_source"]
            # Static-inline read/recall tasks (rainbow/caterpillar passages, story
            # recall, cinderella) carry their reference text via the flat fallback,
            # so the registry leaves stimulus_source unset. Tag it doc-text: the
            # reference is redcap/hand-entered plain text (any accompanying image is
            # illustrative, not the reading target), matching metadata.json.
            if stim_source is None and stim_text and reg["speech_type"] in ("read", "recall"):
                stim_source = "doc-text"
            resolved = {
                "instructions": instructions,
                "speech_type": reg["speech_type"],
                "stimulus_text": stim_text,
                "stimulus_source": stim_source,
                "stimulus_asset": reg["stimulus_asset"],
                "stimulus_asset_n_images": reg.get("stimulus_asset_n_images"),
                "instructions_suffix": reg["instructions_suffix"],
            }
            # The registry knows how many recordings/indices each task has, so an
            # impossible instance (a 5th maximum-phonation-time, a sentences-9) is
            # flagged rather than silently accepted.
            if _registry_numbering_status(task, task_name_lower) == "out-of-range":
                _logger.warning(
                    "task_name=%r matches family %r but its recording index is "
                    "outside the known numbering (recording_count=%s); emitting anyway",
                    task_name, task.get("task_id"), task.get("recording_count"),
                )
        elif flat is not None:
            # The registry -- the authority on known tasks -- has no match, so this
            # name is unknown (e.g. a missed space like "rainbowpassage"). We still
            # fall back to a flat-file best guess, but flag that it is a guess.
            _logger.warning(
                "unknown task_name=%r: no match in the task registry; resolved via "
                "flat-file best-guess. Verify the task name.",
                task_name,
            )
            resolved = {
                "instructions": flat["instructions"],
                "speech_type": flat["speech_type"],
                "stimulus_text": flat["stimulus_text"],
                "stimulus_source": None,
                "stimulus_asset": None,
                "stimulus_asset_n_images": None,
                "instructions_suffix": flat["instructions_suffix"],
            }

        if resolved is not None:
            instructions = resolved["instructions"]
            if resolved["instructions_suffix"]:
                instructions = f"{instructions} {resolved['instructions_suffix']}".strip()
            metadata_file["instructions"] = instructions
            metadata_file["stimulus_text"] = resolved["stimulus_text"]
            metadata_file["speech_type"] = resolved["speech_type"]
            if resolved["stimulus_source"]:
                metadata_file["stimulus_source"] = resolved["stimulus_source"]
            if resolved.get("stimulus_asset"):
                metadata_file["stimulus_asset"] = resolved["stimulus_asset"]
            # Present only for an ordered multi-image sequence (stimulus_asset carries
            # a '{n}' placeholder); its ABSENCE documents that stimulus_asset is a
            # single, directly-resolvable image.
            if resolved.get("stimulus_asset_n_images"):
                metadata_file["stimulus_asset_n_images"] = resolved["stimulus_asset_n_images"]
        else:
            # No match in the registry OR the flat file: the sidecar gets empty
            # instructions/stimulus. Log it so build runs surface unknown task
            # names (a registry/flat gap) rather than silently emitting blanks.
            _logger.warning(
                "no task match for recording task_name=%r (record_id=%s); "
                "instructions/stimulus left empty",
                task_name, participant_id,
            )


    metadata_file.update({"audio_channel_count": 1, "audio_sample_rate": "16000"})

    return metadata_file

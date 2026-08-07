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


def _bank(bank_name: str) -> dict:
    return json.loads(_load_stimulus_bank(bank_name))


def _resolve_prompt_ref(task_name: str, prompt_ref: dict) -> str:
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
    bank = _bank(prompt_ref["bank"])
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
        return bank.get("lists", {}).get(version, {}).get(m.group(1), "")

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


def _resolve_task_registry(task_name: str):
    """Resolve a (granular) task name to (task_entry, recording_entry|None) using
    the registry alias_index (exact, else longest normalized-substring), then the
    nested recording whose recording_id equals the normalized task name. Returns
    None when the registry is absent or nothing matches."""
    reg = _load_registry()
    if not reg:
        return None
    n = _norm(task_name)
    best_tid, best_len = None, -1
    for na, na_len, tid in _alias_items():
        if na == n:
            best_tid, best_len = tid, na_len
            break
        if na in n and na_len > best_len:
            best_tid, best_len = tid, na_len
    if best_tid is None:
        return None
    task = reg["tasks"][best_tid]
    rec = None
    for candidate in task.get("recordings", []):
        if _norm(candidate.get("recording_id", "")) == n:
            rec = candidate
            break
    return task, rec


def _registry_questionnaire(prompt_ref, task_name, join_id, questionnaire_lookup):
    """Per-participant stimulus for a questionnaire-join prompt_ref. Returns
    (stimulus_text, speech_type, instructions_suffix) or None."""
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
        return (str(word).strip() if _is_present(word) else "", "elicited", None)
    if select == "field":
        value = row.get(prompt_ref["field"])
        suffix = f"Category: {str(value).strip()}." if _is_present(value) else None
        return ("", "elicited", suffix)
    if select == "color-list":
        n = prompt_ref.get("n", 15)
        colors = [row.get(prompt_ref["field_template"].format(i=i)) for i in range(1, n + 1)]
        colors = [str(c).strip() for c in colors if _is_present(c)]
        return (" ".join(colors), "read", None)
    return None


def _asset_url(prompt_ref, task_name):
    """A commit-pinned raw GitHub URL for the recording's image stimulus, or None.
    Fills `{i}`/`{i:02d}` in prompt_ref['asset_path'] from the trailing index and
    URL-encodes the path (spaces -> %20)."""
    path = prompt_ref.get("asset_path")
    if not path:
        return None
    idx = _trailing_index(task_name)
    if idx is None:
        return None
    return "https://raw.githubusercontent.com/{repo}/{commit}/{path}".format(
        repo=prompt_ref["asset_repo"],
        commit=prompt_ref["asset_commit"],
        path=quote(path.format(i=idx)),
    )


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


def _registry_bids_fields(task, rec, task_name, join_id, questionnaire_lookup):
    """Assemble sidecar fields from a registry match. `stimulus_text` is None when
    the registry cannot produce it (static-inline / image tasks carry no text) so
    the caller can fall back to the flat file. Returns a dict."""
    # Instructions are authoritative only when curated (per-recording, or a
    # curated task-level instruction such as reading-passage). A coarse harvested
    # task-level instruction is NOT authoritative: the flat file's per-key
    # instruction (e.g. diadochokinesis's per-syllable text, loudness v1/v2) is
    # more specific and must win. The caller consults `instructions_authoritative`.
    if rec and rec.get("instructions"):
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
    instructions_suffix = None

    if ptype == "stimulus-bank":
        stimulus_text = _resolve_prompt_ref(task_name, prompt_ref)
        # reading/repeating-sentences text was transcribed from image stimuli (and
        # carries a pinned URL to that image); the lexical banks
        # (harvard/cape-v/repeating-words) are redcap doc text with no asset.
        stimulus_source = "transcribed" if prompt_ref.get("bank") in (
            "reading_passage_bank", "repeating_sentences_bank"
        ) else "doc-text"
        stimulus_asset = _asset_url(prompt_ref, task_name)
    elif ptype == "questionnaire-join":
        joined = _registry_questionnaire(prompt_ref, task_name, join_id, questionnaire_lookup)
        if joined is not None:
            stimulus_text, speech_type, instructions_suffix = joined
            stimulus_source = "questionnaire"
        else:
            stimulus_text = ""
            stimulus_source = "questionnaire"
    elif ptype == "image":
        # The stimulus is a shown image, not text. Mark provenance and, when the
        # prompt_ref carries a per-index asset path, pin it to a commit-stable URL.
        stimulus_text = ""
        stimulus_source = "image"
        stimulus_asset = _asset_url(prompt_ref, task_name)
    # static-inline: stimulus_text stays None -> flat-file fallback.

    return {
        "instructions": instructions,
        "instructions_authoritative": instructions_authoritative,
        "speech_type": speech_type,
        "stimulus_text": stimulus_text,
        "stimulus_source": stimulus_source,
        "stimulus_asset": stimulus_asset,
        "instructions_suffix": instructions_suffix,
    }


def _flat_bids_fields(task_name_lower, audio_task_descriptions, join_id, questionnaire_lookup):
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

    fields = {
        "instructions": description["instructions"],
        "speech_type": description.get("speech_type") or _classify_speech_type(best_task),
        "stimulus_text": "",
        "instructions_suffix": None,
    }
    prompt_ref = description.get("prompt_ref")
    if prompt_ref:
        fields["stimulus_text"] = _resolve_prompt_ref(task_name_lower, prompt_ref)
    elif "stimulus_text" in description:
        fields["stimulus_text"] = description["stimulus_text"]
    else:
        static_prompts = description.get("prompts", [])
        fields["stimulus_text"] = (
            static_prompts[0] if len(static_prompts) == 1 else " ".join(static_prompts)
        )

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
            task_name_lower, audio_task_descriptions, join_id, questionnaire_lookup
        )
        reg_match = _resolve_task_registry(task_name_lower)

        resolved = None
        if reg_match is not None:
            task, rec = reg_match
            reg = _registry_bids_fields(
                task, rec, task_name_lower, join_id, questionnaire_lookup
            )
            # Registry instructions win only when authoritative (curated);
            # otherwise the flat file's per-key instruction is more specific.
            if reg["instructions_authoritative"] and reg["instructions"]:
                instructions = reg["instructions"]
            elif flat and flat["instructions"]:
                instructions = flat["instructions"]
            else:
                instructions = reg["instructions"]
            resolved = {
                "instructions": instructions,
                "speech_type": reg["speech_type"],
                # registry stimulus_text is None for static-inline/image tasks it
                # does not carry -> fall back to the flat file's text.
                "stimulus_text": reg["stimulus_text"]
                if reg["stimulus_text"] is not None
                else (flat["stimulus_text"] if flat else ""),
                "stimulus_source": reg["stimulus_source"],
                "stimulus_asset": reg["stimulus_asset"],
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

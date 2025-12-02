#!/usr/bin/env python3
"""Generate bids_field_organization.csv from ReproSchema + REDCap metadata."""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tenacity import AsyncRetrying, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import APITimeoutError, APIConnectionError, RateLimitError, APIStatusError
from openai import AsyncOpenAI

# Build a retryable exception tuple that works across openai versions
try:
    _RETRY_EXCEPTIONS = (APITimeoutError, APIConnectionError, RateLimitError, APIStatusError)
except Exception:  # pragma: no cover
    try:
        from openai import APIError  # older versions
        _RETRY_EXCEPTIONS = (APIError,)
    except Exception:
        _RETRY_EXCEPTIONS = (Exception,)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

LOGGER = logging.getLogger("generate_bids_field_organization")

DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
DEFAULT_API_BASE = "https://api.openai.com/v1/"
DEFAULT_PREAMBLE = (
    "You are helping create plain-language descriptions for a BIDS phenotype data "
    "dictionary that will accompany the Bridge2AI Voice dataset. Summaries must be "
    "concise (preferably one sentence), describe what the field captures, mention any "
    "important response options or units, and avoid referencing REDCap internals."
)

SCHEMA_PREFIXES_TO_STRIP = [
    "d_voice_",
    "d_mood_",
    "d_resp_",
    "d_neuro_",
    "q_generic_",
    "q_voice_",
    "q_mood_",
    "q_resp_",
    "q_neuro_",
    "pediatric_q_",
    "pediatric_d_",
]

RENAME_RULES = [
    ("_gsd", "_gold_standard_diagnosis"),
]

INSTRUCTIONS = "You write plain-language data dictionary descriptions for clinical data."


def parse_args() -> argparse.Namespace:
    default_schema = (
        REPO_ROOT
        / "b2ai-redcap2rs"
        / "b2ai-redcap2rs"
        / "b2ai-redcap2rs_schema"
    )
    default_dictionary = REPO_ROOT / "bridge2ai_voice_project_data_dictionary.csv"
    default_output = (
        REPO_ROOT
        / "src"
        / "b2aiprep"
        / "prepare"
        / "resources"
        / "bids_field_organization.csv"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Generate the bids_field_organization.csv file using ReproSchema activities "
            "and the REDCap data dictionary."
        )
    )
    parser.add_argument(
        "--schema-file",
        type=Path,
        default=default_schema,
        help="Path to the root ReproSchema protocol file.",
    )
    parser.add_argument(
        "--data-dictionary",
        type=Path,
        default=default_dictionary,
        help="Path to the REDCap data dictionary CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Destination path for bids_field_organization.csv.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenAI model to use for description generation.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help="Base URL for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--preamble-file",
        type=Path,
        help="Optional file containing a custom prompt preamble.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip OpenAI calls and fall back to question text.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file without prompting.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent OpenAI requests.",
    )
    return parser.parse_args()


def load_redcap_dictionary(dictionary_path: Path) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    with dictionary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            variable = (row.get("Variable / Field Name") or "").strip()
            if not variable:
                continue
            if variable in mapping:
                LOGGER.warning("Duplicate REDCap field '%s' found; keeping the first entry.", variable)
                continue
            mapping[variable] = {k: (v or "").strip() for k, v in row.items()}
    LOGGER.info("Loaded %d REDCap field definitions.", len(mapping))
    return mapping


def normalize_schema_name(activity_id: str) -> str:
    for prefix in SCHEMA_PREFIXES_TO_STRIP:
        if activity_id.startswith(prefix):
            return activity_id[len(prefix) :]
    return activity_id


def build_schema_name_map(activity_ids: Iterable[str]) -> Dict[str, str]:
    provisional = {schema: normalize_schema_name(schema) for schema in activity_ids}
    counts = Counter(provisional.values())
    resolved = {}
    for schema, simplified in provisional.items():
        resolved[schema] = simplified if counts[simplified] == 1 else schema
    return resolved


def rename_column(column_name: str) -> str:
    renamed = column_name
    for target, replacement in RENAME_RULES:
        if target in renamed:
            renamed = renamed.replace(target, replacement)
    return renamed


def format_redcap_metadata(row: Dict[str, str]) -> str:
    ordered_fields = [
        ("Form Name", "Form"),
        ("Section Header", "Section"),
        ("Field Type", "Field Type"),
        ("Field Label", "Prompt"),
        ("Choices, Calculations, OR Slider Labels", "Choices"),
        ("Field Note", "Notes"),
        ("Branching Logic (Show field only if...)" , "Branching Logic"),
        ("Required Field?", "Required"),
        ("Text Validation Type OR Show Slider Number", "Validation"),
        ("Text Validation Min", "Validation Min"),
        ("Text Validation Max", "Validation Max"),
        ("Field Annotation", "Annotation"),
    ]
    parts = []
    for column, label in ordered_fields:
        value = (row.get(column) or "").strip()
        if value:
            parts.append(f"{label}: {value}")
    return "\n".join(parts)


def build_prompt(
    schema_name: str,
    column_name: str,
    metadata_text: str,
    question_text: str,
    preamble: str,
) -> str:
    redcap_text = metadata_text or "No REDCap metadata was found for this field."
    question = question_text or "Question text is not available."
    prompt = textwrap.dedent(
        f"""
        {preamble}

        Schema: {schema_name}
        Field: {column_name}

        REDCap metadata:
        {redcap_text}

        Question text:
        {question}

        Provide a single concise sentence that can serve as the description for this field in a BIDS data dictionary.
        """
    ).strip()
    return prompt

async def _call_openai_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    prompt: str,
    model: str,
) -> str:
    async for attempt in AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    ):
        with attempt:
            async with semaphore:
                response = await client.responses.create(
                    model=model,
                    instructions=INSTRUCTIONS,
                    input=prompt,
                )
        text = (response.output_text or "").strip()
        return text


async def describe_fields_async(
    pending_requests: List[Dict[str, object]],
    model: str,
    api_key: str,
    api_base: str,
    max_concurrency: int,
    description_cache: Dict[str, str],
) -> None:
    if not pending_requests:
        return

    client = AsyncOpenAI(api_key=api_key) # TODO: get api_base to work
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def process_request(request: Dict[str, object]) -> None:
        column_name = request["column_name"]
        prompt = request["prompt"]
        entries = request["entries"]
        description = ""
        try:
            description = await _call_openai_async(client, semaphore, prompt, model)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.error("Failed to generate description for '%s': %s", column_name, exc)

        if description:
            description_cache[column_name] = description

        for entry in entries:
            entry["description"] = description or entry["fallback"]

    await asyncio.gather(*(process_request(req) for req in pending_requests))

def ensure_api_key(dry_run: bool) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key and not dry_run:
        raise RuntimeError("OPENAI_API_KEY is not set. Provide a key or use --dry-run.")
    return api_key


def maybe_load_preamble(path: Optional[Path]) -> str:
    if path:
        return path.read_text(encoding="utf-8").strip()
    return DEFAULT_PREAMBLE


def load_reproschema(schema_file: Path) -> Dict[str, Dict[str, object]]:
    from b2aiprep.prepare.dataset import BIDSDataset  # type: ignore

    return BIDSDataset._load_reproschema(schema_file)


async def run_async(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"Output file {args.output} already exists. Use --force to overwrite."
        )

    api_key = ensure_api_key(args.dry_run)
    preamble = maybe_load_preamble(args.preamble_file)

    LOGGER.info("Loading ReproSchema activities from %s", args.schema_file)
    activities = load_reproschema(args.schema_file)
    schema_name_map = build_schema_name_map(activities.keys())

    redcap_lookup = load_redcap_dictionary(args.data_dictionary)

    description_cache: Dict[str, str] = {}
    row_entries: List[Dict[str, object]] = []
    pending_by_column: Dict[str, List[Dict[str, object]]] = {}
    column_prompts: Dict[str, str] = {}

    for activity_id, payload in activities.items():
        schema_name = schema_name_map[activity_id]
        data_elements = payload.get("data_elements", {})
        LOGGER.info("Processing %s (%d fields)", schema_name, len(data_elements))

        for column_name, element in data_elements.items():
            metadata_row = redcap_lookup.get(column_name)
            if metadata_row is None:
                LOGGER.warning(
                    "No REDCap dictionary entry found for '%s'. Falling back to question text.",
                    column_name,
                )
            metadata_text = format_redcap_metadata(metadata_row) if metadata_row else ""
            question_text = ""
            question = element.get("question")
            if isinstance(question, dict):
                question_text = question.get("en", "").strip()
            elif isinstance(question, str):
                question_text = question.strip()

            renamed_column = rename_column(column_name)
            fallback_description = question_text or (metadata_row or {}).get("Field Label", "")

            entry = {
                "schema_name": schema_name,
                "column_name": column_name,
                "renamed_column": renamed_column,
                "description": None,
                "fallback": fallback_description,
            }
            row_entries.append(entry)

            if args.dry_run:
                entry["description"] = fallback_description
                continue

            cached = description_cache.get(column_name)
            if cached:
                entry["description"] = cached
                continue

            if column_name not in column_prompts:
                column_prompts[column_name] = build_prompt(
                    schema_name=schema_name,
                    column_name=column_name,
                    metadata_text=metadata_text,
                    question_text=question_text,
                    preamble=preamble,
                )
            pending_by_column.setdefault(column_name, []).append(entry)

    if not args.dry_run and pending_by_column:
        pending_requests = [
            {
                "column_name": column,
                "prompt": column_prompts[column],
                "entries": pending_by_column[column],
            }
            for column in column_prompts.keys()
        ]
        await describe_fields_async(
            pending_requests=pending_requests,
            model=args.model,
            api_key=api_key,
            api_base=args.api_base,
            max_concurrency=args.max_concurrency,
            description_cache=description_cache,
        )

    for entry in row_entries:
        entry["description"] = entry.get("description") or entry["fallback"]

    rows = [
        {
            "schema_name": entry["schema_name"],
            "column_name": entry["column_name"],
            "renamed_column": entry["renamed_column"],
            "description": entry["description"],
        }
        for entry in row_entries
    ]

    rows.sort(key=lambda row: (row["schema_name"], row["column_name"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["schema_name", "column_name", "renamed_column", "description"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    LOGGER.info("Wrote %d rows to %s", len(rows), args.output)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()

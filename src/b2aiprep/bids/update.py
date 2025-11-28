"""Utilities for regenerating BIDS template metadata from reproschema sources."""
from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

_LOGGER = logging.getLogger(__name__)

_RAW_BASE_URL = "https://raw.githubusercontent.com/sensein/b2ai-redcap2rs"


class TemplateUpdateError(RuntimeError):
    """Raised when the template update process cannot be completed."""


def populate_description(_: str) -> str:
    """Placeholder hook for populating activity descriptions."""

    return ""


def populate_data_element_description(_: str) -> str:
    """Placeholder hook for populating item-level descriptions."""

    return ""


def _read_json(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise TemplateUpdateError(f"Failed to parse {path}: {exc}") from exc


def _slugify_filename(label: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", label).strip("_")
    return slug or "template"


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _get_commit_sha(submodule_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(submodule_root), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise TemplateUpdateError(
            f"Unable to determine commit for submodule at {submodule_root}"
        ) from exc
    return result.stdout.strip()


def _build_raw_url(commit_sha: str, relative_path: Path) -> str:
    return f"{_RAW_BASE_URL}/{commit_sha}/{relative_path.as_posix()}"


def _build_data_elements(
    activity_json: Dict,
    activity_path: Path,
    submodule_root: Path,
    commit_sha: str,
) -> Dict[str, Dict]:
    order = activity_json.get("ui", {}).get("order", [])
    data_elements: Dict[str, Dict] = {}

    for item_rel_path in order:
        item_path = (activity_path.parent / item_rel_path).resolve()
        if not item_path.exists():
            _LOGGER.warning("Skipping missing item path %s", item_rel_path)
            continue

        try:
            item_rel_to_repo = item_path.relative_to(submodule_root)
        except ValueError as exc:
            raise TemplateUpdateError(
                f"Item path {item_path} is outside submodule {submodule_root}"
            ) from exc

        item_json = _read_json(item_path)
        element_id = item_json.get("id") or Path(item_rel_path).stem
        if not element_id:
            raise TemplateUpdateError(f"Unable to derive identifier for {item_path}")

        response_options = item_json.get("responseOptions", {}) or {}
        value_type = response_options.get("valueType")
        datatype = response_options.get("datatype") or value_type

        element_payload: Dict[str, Optional[object]] = {
            "description": populate_data_element_description(element_id),
            "question": item_json.get("question"),
            "datatype": datatype,
            "choices": response_options.get("choices"),
            "termURL": _build_raw_url(commit_sha, item_rel_to_repo),
            "valueType": value_type,
        }

        for key, value in response_options.items():
            if key in {"choices", "valueType", "datatype"}:
                continue
            element_payload[key] = value

        data_elements[element_id] = element_payload

    return data_elements


def _build_activity_payload(
    activity_json: Dict,
    activity_path: Path,
    submodule_root: Path,
    commit_sha: str,
) -> Dict:
    activity_id = activity_json.get("id")
    if not activity_id:
        raise TemplateUpdateError(f"Activity file {activity_path} is missing an id")

    try:
        activity_rel_path = activity_path.relative_to(submodule_root)
    except ValueError as exc:
        raise TemplateUpdateError(
            f"Activity path {activity_path} is outside submodule {submodule_root}"
        ) from exc

    return {
        "description": populate_description(activity_id),
        "url": _build_raw_url(commit_sha, activity_rel_path),
        "data_elements": _build_data_elements(
            activity_json=activity_json,
            activity_path=activity_path,
            submodule_root=submodule_root,
            commit_sha=commit_sha,
        ),
    }


def update_bids_template_files(
    submodule_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    schema_file: Optional[Path] = None,
    dry_run: bool = False,
) -> List[Path]:
    """Generate phenotype template JSON files from the reproschema definition.

    Args:
        submodule_path: Optional override path to the cloned b2ai-redcap2rs repo.
        output_dir: Destination directory for generated JSON files.
        schema_file: Optional override for the protocol schema file.
        dry_run: When True, compute outputs without writing files.

    Returns:
        A list containing the output file paths (real or planned when dry_run is True).
    """

    repo_root = _get_repo_root()
    submodule_root = (
        Path(submodule_path) if submodule_path else repo_root / "b2ai-redcap2rs"
    ).resolve()
    if not submodule_root.exists():
        raise TemplateUpdateError(f"Submodule directory {submodule_root} does not exist")

    resolved_schema_file = (
        Path(schema_file)
        if schema_file
        else submodule_root / "b2ai-redcap2rs" / "b2ai-redcap2rs_schema"
    ).resolve()
    if not resolved_schema_file.exists():
        raise TemplateUpdateError(f"Schema file {resolved_schema_file} was not found")

    output_dir_path = (
        Path(output_dir) if output_dir else repo_root / "src/b2aiprep/template/phenotype"
    ).resolve()
    if not dry_run:
        output_dir_path.mkdir(parents=True, exist_ok=True)

    schema_json = _read_json(resolved_schema_file)
    protocol_order = schema_json.get("ui", {}).get("order")
    if not protocol_order:
        raise TemplateUpdateError("Protocol schema does not define ui.order entries")

    commit_sha = _get_commit_sha(submodule_root)

    generated_files: List[Path] = []
    for rel_path in protocol_order:
        activity_path = (resolved_schema_file.parent / rel_path).resolve()
        if not activity_path.exists():
            _LOGGER.warning("Skipping missing activity %s", rel_path)
            continue

        activity_json = _read_json(activity_path)
        activity_id = activity_json.get("id")
        if not activity_id:
            raise TemplateUpdateError(f"Activity {activity_path} missing id")

        pref_label = (activity_json.get("prefLabel") or {}).get("en") or activity_id
        file_stem = _slugify_filename(pref_label)
        output_file = (output_dir_path / f"{file_stem}.json").resolve()
        generated_files.append(output_file)

        payload = _build_activity_payload(
            activity_json=activity_json,
            activity_path=activity_path,
            submodule_root=submodule_root,
            commit_sha=commit_sha,
        )

        if dry_run:
            continue

        with output_file.open("w", encoding="utf-8") as handle:
            json.dump({activity_id: payload}, handle, indent=4)
            handle.write("\n")

        _LOGGER.info("Wrote %s", output_file)

    return generated_files

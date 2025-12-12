"""Utilities for regenerating BIDS template metadata from reproschema sources."""
from __future__ import annotations

from copy import copy
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
from importlib.resources import files

import pandas as pd

from b2aiprep.prepare.utils import get_commit_sha

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


def _slugify_filename(label: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", label).strip("_")
    return slug or "template"


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _build_raw_url(commit_sha: str, relative_path: Path) -> str:
    return f"{_RAW_BASE_URL}/{commit_sha}/{relative_path.as_posix()}"


def _build_data_elements(
    activity_json: Dict,
    activity_path: Path,
    reproschema_folder: Path,
    commit_sha: str,
) -> Dict[str, Dict]:
    order = activity_json.get("ui", {}).get("order", [])
    data_elements: Dict[str, Dict] = {}

    # add in additional items that may only be in addProperties,
    # e.g. calculated fields
    for additional_item in activity_json.get("ui", {}).get("addProperties", []):
        item_name = additional_item.get("isAbout", None)
        if item_name and item_name.startswith("items/") and item_name not in order:
            order.append(item_name)

    for item_rel_path in order:
        item_path = (activity_path.parent / item_rel_path).resolve()
        if not item_path.exists():
            _LOGGER.warning("Skipping missing item path %s", item_rel_path)
            continue

        try:
            item_rel_to_repo = item_path.relative_to(reproschema_folder)
        except ValueError as exc:
            raise TemplateUpdateError(
                f"Item path {item_path} is outside path {reproschema_folder}"
            ) from exc

        item_json = json.loads(item_path.read_text())
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


def build_activity_payload(
    activity_json: Dict,
    activity_path: Path,
    reproschema_folder: Path,
    commit_sha: str,
) -> Dict:
    activity_id = activity_json.get("id")
    if not activity_id:
        raise TemplateUpdateError(f"Activity file {activity_path} is missing an id")

    try:
        activity_rel_path = activity_path.relative_to(reproschema_folder)
    except ValueError as exc:
        raise TemplateUpdateError(
            f"Activity path {activity_path} is outside submodule {reproschema_folder}"
        ) from exc

    return {
        "description": populate_description(activity_id),
        "url": _build_raw_url(commit_sha, activity_rel_path),
        "data_elements": _build_data_elements(
            activity_json=activity_json,
            activity_path=activity_path,
            reproschema_folder=reproschema_folder,
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

    schema_json = json.loads(resolved_schema_file.read_text())
    protocol_order = schema_json.get("ui", {}).get("order")
    if not protocol_order:
        raise TemplateUpdateError("Protocol schema does not define ui.order entries")

    commit_sha = get_commit_sha(submodule_root)

    generated_files: List[Path] = []
    for rel_path in protocol_order:
        activity_path = (resolved_schema_file.parent / rel_path).resolve()
        if not activity_path.exists():
            _LOGGER.warning("Skipping missing activity %s", rel_path)
            continue

        activity_json = json.loads(activity_path.read_text())
        activity_id = activity_json.get("id")
        if not activity_id:
            raise TemplateUpdateError(f"Activity {activity_path} missing id")

        pref_label = (activity_json.get("prefLabel") or {}).get("en") or activity_id
        file_stem = _slugify_filename(pref_label)
        output_file = (output_dir_path / f"{file_stem}.json").resolve()
        generated_files.append(output_file)

        payload = build_activity_payload(
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

    return generated_files



def reorganize_bids_activities(
    template_dir: Path, dry_run: bool = False
) -> List[Path]:
    """Reorganizes a set of phenotype template JSON files.

    Args:
        template_dir: Source and destination directory for generated JSON files.

    Returns:
        A list containing the output file paths (real or planned when dry_run is True).
    """
    existing_files = sorted(list(template_dir.glob('*.json')))

    # load in the existing structures
    # since all the schema JSONs are nested under the schema id - e.g. {schema_name: {... }},
    # we can load them all into the same dict.
    schemas = {}
    element_to_schema = {} # index for data_element -> schema
    for json_file in existing_files:
        with open(json_file, 'r') as fp:
            json_data = json.load(fp)
        schema_name = next(iter(json_data))
        for element_name in json_data[schema_name]['data_elements'].keys():
            element_to_schema[element_name] = schema_name
        schemas.update(json_data)
    
    reorganization_file = files("b2aiprep.prepare.resources").joinpath("bids_field_organization.csv")
    df_reorg = pd.read_csv(reorganization_file, sep=',', header=0)

    updated_schemas = {}
    element_used = {} # keep track of whether we have used an element, for logging later.
    for activity_id, group in df_reorg.groupby('schema_name'):
        column_mapping = group.set_index('column_name').to_dict(orient='index')

        payload = {
            "description": "",
            "data_elements": {}
        }
        for column, updated_data in column_mapping.items():
            schema_to_use = element_to_schema[column]
            data_element = copy(schemas[schema_to_use]["data_elements"][column])

            if "description" in updated_data and (updated_data["description"] != ""):
                description = updated_data["description"]
            elif "description" in data_element and (data_element["description"] != ""):
                description = data_element["description"]
            else:
                description = data_element.get("question", "").get("en", "")
            data_element["description"] = description
            new_element_name = updated_data["renamed_column"]
            payload["data_elements"][new_element_name] = data_element
            element_used[column] = True

        updated_schemas[activity_id] = payload

    # now we will write out & clean up the old template files
    files_to_remove = [
        json_file for json_file in existing_files
        if json_file.stem not in updated_schemas
    ]
    files_saved = []
    for activity_id, payload in updated_schemas.items():
        output_file = template_dir.joinpath(f'{activity_id}.json')
        files_saved.append(output_file)

        if dry_run:
            continue

        with output_file.open("w", encoding="utf-8") as fp:
            json.dump({activity_id: payload}, fp, indent=4)
            fp.write("\n")
        
        for file in files_to_remove:
            file.unlink()

    _LOGGER.info(f"Reorganized BIDS phenotype templates into {len(updated_schemas)} files.")
    elements_skipped = [elem for elem in element_to_schema if elem not in element_used]
    _LOGGER.info(f"Reorganization ignored {len(elements_skipped)} columns: {elements_skipped}")
    return files_saved

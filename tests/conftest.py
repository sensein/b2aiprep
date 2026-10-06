from pathlib import Path
import json

import pytest

@pytest.fixture(scope="module")
def reproschema_module_path():
    project_root = Path(__file__).parent.parent
    reproschema_path = project_root.joinpath("b2ai-redcap2rs", "b2ai-redcap2rs").resolve().as_posix()
    return reproschema_path



@pytest.fixture
def setup_publish_config(tmp_path):
    """Fixture to create a publish config directory with default empty files."""
    config_dir = tmp_path / "publish_config"
    config_dir.mkdir()

    # Default empty configurations
    defaults = {
        "audio_filestems_to_remove.json": [],
        "id_remapping.json": {},
        "participants_to_remove.json": [],
        "audio_tasks_to_include.json": ["test"],
        "deidentify_settings.json": {"access_tier": "registered"},
    }

    for filename, content in defaults.items():
        with open(config_dir / filename, "w") as f:
            json.dump(content, f, indent=2)

    return config_dir



WAV_BYTES = b"RIFF" + b"\x00" * 8192  # enough for deidentify to copy; never decoded


def write_phenotype_table(folder, name, df, choices=None):
    """Write phenotype/<...>/<name>.tsv and its sidecar; *choices* maps column -> answer labels."""
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f"{name}.tsv", sep="\t", index=False)
    elements = {c: {"description": c} for c in df.columns}
    for column, labels in (choices or {}).items():
        elements[column]["choices"] = [{"name": {"en": label}, "value": label.lower()} for label in labels]
    (folder / f"{name}.json").write_text(json.dumps({name: {"description": name, "data_elements": elements}}))


@pytest.fixture
def make_deid_tree(tmp_path):
    """Build a small BIDS tree and deidentify config; returns ``make(...) -> (bids, config)``.

    ``recordings``: ``{participant: [(session, task) or (session, task, {extra sidecar keys}), ...]}``.
    ``sessions``: ``{participant: [session, ...]}`` for sessions.tsv (default: the recordings'
    sessions, numbered 1..n in that order). ``tables``: ``{"group/name" or "name": DataFrame}``;
    ``choices``: ``{"group/name": {column: [labels]}}``. ``settings`` are merged over
    ``{"access_tier": "registered"}``; ``config_files`` add or replace config JSONs. ``pseudonyms``
    default to 900000, 900001, ... in participant order. ``features`` also writes a .pt per recording.
    """
    import pandas as pd

    def make(recordings, *, root=None, sessions=None, tables=None, choices=None, settings=None,
             config_files=None, pseudonyms=None, tasks=None, features=False):
        root = Path(root or tmp_path)
        bids, config = root / "bids", root / "config"
        config.mkdir(parents=True)
        participants = list(recordings)
        used_tasks = sorted({r[1] for recs in recordings.values() for r in recs})
        files = {
            "participants_to_include.json": participants,
            "id_remapping.json": pseudonyms if pseudonyms is not None
            else {p: f"9{i:05d}" for i, p in enumerate(participants)},
            "deidentify_settings.json": {"access_tier": "registered", **(settings or {})},
            "audio_tasks_to_include.json": list(tasks) if tasks is not None else used_tasks,
            "audio_filestems_to_remove.json": [],
            **(config_files or {}),
        }
        for name, obj in files.items():
            (config / name).write_text(json.dumps(obj))
        for pid, recs in recordings.items():
            for rec in recs:
                ses, task, extra = rec[0], rec[1], (rec[2] if len(rec) > 2 else {})
                audio = bids / f"sub-{pid}" / f"ses-{ses}" / "audio"
                audio.mkdir(parents=True, exist_ok=True)
                stem = f"sub-{pid}_ses-{ses}_task-{task}"
                (audio / f"{stem}.wav").write_bytes(WAV_BYTES)
                (audio / f"{stem}_recording-metadata.json").write_text(
                    json.dumps({"record_id": pid, "session_id": ses, **extra}))
                if features:
                    import torch
                    torch.save({"opensmile": {"x": 1}}, audio / f"{stem}_features.pt")
            ses_ids = (sessions or {}).get(pid)
            if ses_ids is None:
                ses_ids = list(dict.fromkeys(r[0] for r in recs))
            if isinstance(ses_ids, pd.DataFrame):
                ses_df = ses_ids
            else:
                ses_df = pd.DataFrame({"record_id": [pid] * len(ses_ids), "session_id": ses_ids,
                                       "session_index": [str(i) for i in range(1, len(ses_ids) + 1)]})
            ses_df.to_csv(bids / f"sub-{pid}" / "sessions.tsv", sep="\t", index=False)
        (bids / "phenotype").mkdir(parents=True, exist_ok=True)
        for path, df in (tables or {}).items():
            group, _, name = path.rpartition("/")
            write_phenotype_table(bids / "phenotype" / group if group else bids / "phenotype", name, df,
                                  (choices or {}).get(path))
        (bids / "dataset_description.json").write_text(json.dumps({"Name": "test"}))
        return bids, config

    return make

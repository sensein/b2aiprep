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
        "sensitive_audio_tasks.json": [],
    }

    for filename, content in defaults.items():
        with open(config_dir / filename, "w") as f:
            json.dump(content, f, indent=2)

    return config_dir


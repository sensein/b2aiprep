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
    }

    for filename, content in defaults.items():
        with open(config_dir / filename, "w") as f:
            json.dump(content, f, indent=2)

    return config_dir


# ---------------------------------------------------------------------------
# Synthetic WAV fixture factory (used by T009-T013)
# ---------------------------------------------------------------------------


def create_dummy_wav_file(
    tmp_dir: Path,
    wav_type: str = "clean",
    filename: str = None,
    sample_rate: int = 16000,
    duration_s: float = 2.0,
) -> Path:
    """Create a synthetic WAV file of the requested type.

    Uses ``torchaudio.save`` so the output is compatible with torchaudio
    loaders used throughout the pipeline.

    Args:
        tmp_dir:     Directory to write the file into.
        wav_type:    One of ``"clean"``, ``"clipped"``, ``"silent"``, ``"noisy"``.
        filename:    Output filename.  Defaults to ``"{wav_type}.wav"``.
        sample_rate: Sample rate in Hz (default 16 000).
        duration_s:  Duration in seconds (default 2.0).

    Returns:
        Path to the written WAV file.
    """
    import numpy as np
    import torch
    import torchaudio

    n = int(sample_rate * duration_s)
    rng = np.random.default_rng(42)

    if wav_type == "clean":
        t = np.linspace(0, duration_s, n, endpoint=False)
        signal = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    elif wav_type == "clipped":
        # > 5 % of samples at full scale → proportion_clipped > clipping_max (0.05)
        signal = np.ones(n, dtype=np.float32)
    elif wav_type == "silent":
        # All zeros → proportion_silent > silence_max (0.50)
        signal = np.zeros(n, dtype=np.float32)
    elif wav_type == "noisy":
        # White noise — low SNR environment
        signal = rng.uniform(-0.9, 0.9, n).astype(np.float32)
    else:
        raise ValueError(f"Unknown wav_type: {wav_type!r}")

    fname = filename or f"{wav_type}.wav"
    out_path = tmp_dir / fname
    waveform = torch.from_numpy(signal).unsqueeze(0)  # shape [1, n_samples]
    torchaudio.save(str(out_path), waveform, sample_rate, format="wav")
    return out_path


@pytest.fixture
def wav_factory(tmp_path):
    """Pytest fixture returning the ``create_dummy_wav_file`` factory.

    By default writes into the test's ``tmp_path``.  Pass ``target_dir`` to
    write into a specific subdirectory (e.g. a synthetic BIDS voice folder).
    """

    def _factory(
        wav_type="clean",
        filename=None,
        sample_rate=16000,
        duration_s=2.0,
        target_dir=None,
    ):
        d = Path(target_dir) if target_dir is not None else tmp_path
        return create_dummy_wav_file(
            d,
            wav_type=wav_type,
            filename=filename,
            sample_rate=sample_rate,
            duration_s=duration_s,
        )

    return _factory

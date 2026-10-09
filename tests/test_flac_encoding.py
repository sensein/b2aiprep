"""Released audio is encoded as 16-bit FLAC by one pinned encoder (soundfile/libsndfile), so the
same samples always produce the same bytes and Synapse does not see unchanged audio as updated."""
import hashlib

import numpy as np
import soundfile as sf
import torch
from senselab.audio.data_structures.audio import Audio

from b2aiprep.prepare.dataset import _copy_audio_files_parallel, _save_as_flac


def _pcm16_source(path, channels=1, seconds=1.0, rate=16000):
    rng = np.random.default_rng(0)
    samples = rng.integers(-30000, 30000, size=(int(seconds * rate), channels), dtype=np.int16)
    sf.write(path, samples, rate, subtype="PCM_16")
    return samples


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_flac_is_16_bit(tmp_path):
    src = tmp_path / "src.wav"
    _pcm16_source(src)
    dst = tmp_path / "out.flac"
    _save_as_flac(Audio(filepath=str(src)), dst)
    info = sf.info(dst)
    assert info.format == "FLAC"
    assert info.subtype == "PCM_16"


def test_flac_bytes_are_reproducible(tmp_path):
    src = tmp_path / "src.wav"
    _pcm16_source(src)
    first, second = tmp_path / "a.flac", tmp_path / "b.flac"
    _save_as_flac(Audio(filepath=str(src)), first)
    _save_as_flac(Audio(filepath=str(src)), second)
    assert _sha256(first) == _sha256(second)


def test_copy_without_sanitize_is_lossless(tmp_path):
    src = tmp_path / "src.wav"
    samples = _pcm16_source(src, channels=2)
    dst = tmp_path / "out.flac"
    _copy_audio_files_parallel([(src, dst)], max_workers=1, sanitize_audio_format=False)
    decoded, rate = sf.read(dst, dtype="int16", always_2d=True)
    assert rate == 16000
    assert np.array_equal(decoded, samples)


def test_sanitize_overshoot_is_guarded_not_clipped(tmp_path):
    # A source peaking just under full scale: resampling 48 kHz -> 16 kHz overshoots it, and an
    # unguarded PCM_16 write would clip that overshoot to full scale.
    rate = 48000
    t = np.arange(rate) / rate
    square = np.sign(np.sin(2 * np.pi * 440 * t)) * 0.98
    src = tmp_path / "src.wav"
    sf.write(src, square.astype(np.float32), rate, subtype="FLOAT")
    dst = tmp_path / "out.flac"
    _copy_audio_files_parallel([(src, dst)], max_workers=1, sanitize_audio_format=True)
    assert sf.info(dst).subtype == "PCM_16"
    decoded, out_rate = sf.read(dst, dtype="int16")
    assert out_rate == 16000
    assert np.abs(decoded.astype(np.int32)).max() < 32767  # nothing pinned at full scale


def test_unguarded_overshoot_would_clip(tmp_path):
    """Why the guard must stay in front of _save_as_flac: libsndfile clips |x| > 1 silently."""
    dst = tmp_path / "hot.flac"
    _save_as_flac(Audio(waveform=torch.tensor([[0.0, 1.047, -1.03]]), sampling_rate=16000), dst)
    decoded, _ = sf.read(dst, dtype="int16")
    assert list(decoded[1:]) == [32767, -32768]

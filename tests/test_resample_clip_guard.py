"""The audio-copy resample step must not let the 16-bit save silently clamp
overshoot; _guard_resample_overshoot rescales to the input peak instead."""
import torch
import soundfile as sf
from senselab.audio.data_structures.audio import Audio
from b2aiprep.prepare.dataset import PCM16_MAX, _guard_resample_overshoot, _save_as_flac


def test_overshoot_is_rescaled_not_clamped(tmp_path):
    # a waveform that exceeds full-scale (simulating resample overshoot) on a
    # source whose own peak was below 1.0
    wf = torch.tensor([[0.0, 0.5, -0.98, 1.047, -1.03, 0.2]])
    a = Audio(waveform=wf, sampling_rate=16000)
    guarded, scale = _guard_resample_overshoot(a, in_peak=0.98)
    assert scale is not None and 0 < scale < 1
    assert float(guarded.waveform.abs().max()) <= 1.0
    # peak restored to the input peak (gain preserved), not clamped to 1.0
    assert abs(float(guarded.waveform.abs().max()) - 0.98) < 1e-6
    # and it actually avoids clipping through the real 16-bit FLAC write
    dst = tmp_path / "g.flac"
    _save_as_flac(guarded, dst)
    x, _ = sf.read(str(dst))
    assert (abs(x) >= 0.9995).mean() == 0.0


def test_in_range_is_noop():
    wf = torch.tensor([[0.0, 0.5, -0.9, 0.99]])
    a = Audio(waveform=wf, sampling_rate=16000)
    guarded, scale = _guard_resample_overshoot(a, in_peak=0.99)
    assert scale is None
    assert guarded is a


def test_full_scale_peak_fits_16_bit():
    # +1.0 maps to 32768, one past the largest 16-bit sample, so it is scaled to PCM16_MAX
    a = Audio(waveform=torch.tensor([[0.0, 1.0, -0.5]]), sampling_rate=16000)
    guarded, scale = _guard_resample_overshoot(a, in_peak=1.0)
    assert scale is not None
    assert abs(float(guarded.waveform.abs().max()) - PCM16_MAX) < 1e-7

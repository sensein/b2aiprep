"""The audio-copy resample step must not let the 16-bit save silently clamp
overshoot; _guard_resample_overshoot rescales to the input peak instead."""
import torch
import soundfile as sf
from senselab.audio.data_structures.audio import Audio
from b2aiprep.prepare.dataset import _guard_resample_overshoot


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
    # and it actually avoids clipping through a real 16-bit save round-trip
    dst = tmp_path / "g.wav"
    guarded.save_to_file(str(dst), bits_per_sample=16)
    x, _ = sf.read(str(dst))
    assert (abs(x) >= 0.9995).mean() == 0.0


def test_in_range_is_noop():
    wf = torch.tensor([[0.0, 0.5, -0.9, 0.99]])
    a = Audio(waveform=wf, sampling_rate=16000)
    guarded, scale = _guard_resample_overshoot(a, in_peak=0.99)
    assert scale is None
    assert guarded is a

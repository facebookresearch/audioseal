# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import urllib

import pytest
import torch
import torchaudio

from audioseal import AudioSeal
from audioseal.models import AudioSealDetector, AudioSealWM


@pytest.fixture
def example_audio(tmp_path):
    url = "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav"
    with open(tmp_path / "test.wav", "wb") as f:
        resp = urllib.request.urlopen(url)
        f.write(resp.read())

    wav, sr = torchaudio.load(tmp_path / "test.wav")

    # Add batch dimension
    yield wav.expand(2, 1, -1), sr


def test_detector(example_audio):
    audio, sr = example_audio
    model = AudioSeal.load_generator("audioseal_wm_16bits")
    model.eval()

    secret_message = torch.randint(0, 2, (1, 16), dtype=torch.int32)
    watermark = model.get_watermark(audio, sample_rate=sr, message=secret_message)

    watermarked_audio = audio + watermark

    detector = AudioSeal.load_detector(("audioseal_detector_16bits"))
    detector.eval()
    results, message = detector.detect_watermark(watermarked_audio, sample_rate=sr)  # noqa

    # Due to non-deterministic decoding, messages are not always the same as message
    print(f"\nOriginal message: {secret_message}")
    print(f"Decoded message: {message}")
    print(
        "Matching bits in decoded and original messages: "
        f"{torch.count_nonzero(torch.eq(message, secret_message)).item()}\n"
    )
    assert torch.count_nonzero(torch.eq(message, secret_message)).item() > 20
    assert torch.all(results > 0.5).item()

    # Try to detect the unwatermarked audio
    results, _ = detector.detect_watermark(audio, sample_rate=sr)  # noqa
    assert torch.all(results < 0.5).item()


def test_loading_from_hf():
    generator = AudioSeal.load_generator("facebook/audioseal/generator_base.pth", nbits=16)
    detector = AudioSeal.load_detector("facebook/audioseal/detector_base.pth", nbits=16)

    assert isinstance(generator, AudioSealWM) and isinstance(detector, AudioSealDetector)


def test_jit(example_audio, tmp_path):
    # Test that the audioseal model can be torchscripted
    model = AudioSeal.load_generator("audioseal_wm_16bits")
    model.eval()
    scripted_model = torch.jit.script(model)
    assert isinstance(scripted_model, torch.jit.ScriptModule)
    scripted_model.save(tmp_path / "audioseal_wm_16bits.jit")
    del scripted_model

    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    scripted_detector = torch.jit.script(detector)
    scripted_detector.save(tmp_path / "audioseal_detector_16bits.jit")
    del scripted_detector

    jit_generator = torch.jit.load(tmp_path / "audioseal_wm_16bits.jit")
    jit_detector = torch.jit.load(tmp_path / "audioseal_detector_16bits.jit")
    audio, _ = example_audio

    wm_audio = jit_generator(audio, alpha=0.8)
    result, _ = jit_detector.detect_watermark(wm_audio)
    assert torch.all(result > 0.5).item(), "JIT model failed to detect watermark"


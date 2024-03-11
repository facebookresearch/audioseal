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


@pytest.fixture
def example_audio(tmp_path):
    url = "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav"
    with open(tmp_path / "test.wav", "wb") as f:
        resp = urllib.request.urlopen(url)
        f.write(resp.read())

    wav, sr = torchaudio.load(tmp_path / "test.wav")

    # Add batch dimension
    yield wav.unsqueeze(0), sr


def test_detector(example_audio):
    audio, sr = example_audio
    model = AudioSeal.load_generator("audioseal_wm_16bits")

    secret_message = torch.randint(0, 2, (1, 16), dtype=torch.int32)
    watermark = model(audio, sample_rate=sr, message=secret_message, alpha=0.8)

    watermarked_audio = audio + watermark

    detector = AudioSeal.load_detector(("audioseal_detector_16bits"))
    result, message = detector.detect_watermark(watermarked_audio, sample_rate=sr)   # noqa

    # Due to non-deterministic decoding, messages are not always the same as message
    print(f"\nOriginal message: {secret_message}")
    print(f"Decoded message: {message}")
    print(
        "Matching bits in decoded and original messages: "
        f"{torch.count_nonzero(torch.eq(message, secret_message)).item()}\n"
    )
    assert result > 0.6

    # Try to detect the unwatermarked audio
    result, _ = detector.detect_watermark(audio, sample_rate=sr)  # noqa
    assert result < 0.5

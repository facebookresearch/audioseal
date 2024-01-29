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

    wav, _ = torchaudio.load(tmp_path / "test.wav")

    # Add batch dimension
    wav = wav.unsqueeze(0)

    yield wav


def test_detector(example_audio):
    print(example_audio.size())

    model = AudioSeal.load_generator("audioseal_wm_16bits")

    secret_message = torch.randint(0, 2, (1, 16))
    watermark = model(example_audio, message=secret_message, alpha=0.8)

    watermarked_audio = example_audio + watermark

    detector = AudioSeal.load_detector(("audioseal_detector_16bits"))
    result, message = detector(watermarked_audio)   # noqa

    pred_prob = torch.count_nonzero(torch.gt(result[:, 1, :], 0.5)) / result.shape[-1]

    assert pred_prob.item() > 0.7

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import urllib
from pathlib import Path

import pytest
import torch
import torchaudio

from audioseal import AudioSeal
from audioseal.builder import (
    AudioSealDetectorConfig,
    AudioSealWMConfig,
    create_detector,
    create_generator,
)
from audioseal.models import AudioSealDetector, AudioSealWM
from scripts.combine_checkpoints import combine_checkpoints


@pytest.fixture
def ckpts_dir() -> Path:
    path = Path("TMP")
    path.mkdir(exist_ok=True, parents=True)

    return path


@pytest.fixture
def generator_ckpt_path(ckpts_dir: Path) -> Path:

    checkpoint, config = AudioSeal.parse_model(
        "audioseal_wm_16bits",
        AudioSealWMConfig,
        nbits=16,
    )

    model = create_generator(config)
    model.load_state_dict(checkpoint)

    checkpoint = {"xp.cfg": config, "model": model.state_dict()}
    path = ckpts_dir / "generator_checkpoint.pth"

    torch.save(checkpoint, path)

    return path


@pytest.fixture
def detector_ckpt_path(ckpts_dir: Path) -> Path:

    checkpoint, config = AudioSeal.parse_model(
        "audioseal_detector_16bits",
        AudioSealDetectorConfig,
        nbits=16,
    )

    model = create_detector(config)
    model.load_state_dict(checkpoint)

    checkpoint = {"xp.cfg": config, "model": model.state_dict()}
    path = ckpts_dir / "detector_checkpoint.pth"

    torch.save(checkpoint, path)

    return path


def test_combine_checkpoints(
    generator_ckpt_path: Path, detector_ckpt_path: Path, ckpts_dir: Path
):

    combined_ckpt_path = ckpts_dir / "combined.pth"

    combine_checkpoints(generator_ckpt_path, detector_ckpt_path, combined_ckpt_path)

    assert combined_ckpt_path.exists()

    generator = torch.load(generator_ckpt_path)
    detector = torch.load(detector_ckpt_path)

    combined = torch.load(combined_ckpt_path)

    for key in generator["model"]:
        assert f"generator.{key}" in combined["model"]

    for key in detector["model"]:
        assert f"detector.{key}" in combined["model"]

    # clean up
    combined_ckpt_path.unlink()
    generator_ckpt_path.unlink()
    detector_ckpt_path.unlink()
    ckpts_dir.rmdir()

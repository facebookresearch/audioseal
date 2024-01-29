# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from torch import device, dtype
from typing_extensions import TypeAlias

import audiocraft
from audioseal.models import AudioSealDetector, AudioSealWM, MsgProcessor

Device: TypeAlias = device

DataType: TypeAlias = dtype


@dataclass
class SEANetConfig:
    """
    Map common hparams of SEANet encoder and decoder.
    """

    channels: int
    dimension: int
    n_filters: int
    n_residual_layers: int
    ratios: List[int]
    activation: str
    activation_params: Dict[str, float]
    norm: str
    norm_params: Dict[str, Any]
    kernel_size: int
    last_kernel_size: int
    residual_kernel_size: int
    dilation_base: int
    causal: bool
    pad_mode: str
    true_skip: bool
    compress: int
    lstm: int
    disable_norm_outer_blocks: int


@dataclass
class DecoderConfig:
    final_activation: Optional[str]
    final_activation_params: Optional[dict]
    trim_right_ratio: float


@dataclass
class DetectorConfig:
    output_dim: int


@dataclass
class AudioSealWMConfig:
    nbits: int
    seanet: SEANetConfig
    decoder: DecoderConfig


@dataclass
class AudioSealDetectorConfig:
    nbits: int
    seanet: SEANetConfig
    detector: DetectorConfig


def create_generator(
    config: AudioSealWMConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> AudioSealWM:
    """Create a generator from hparams"""

    #  Currently the encoder hparams are the same as
    # SEANet, but this can be changed in the future.
    encoder = audiocraft.modules.SEANetEncoder(**config.seanet)  # type: ignore[arg-type]
    encoder = encoder.to(device=device, dtype=dtype)

    decoder_config = {**config.seanet, **config.decoder}  # type: ignore
    decoder = audiocraft.modules.SEANetDecoder(**decoder_config)  # type: ignore[arg-type]
    decoder = decoder.to(device=device, dtype=dtype)

    msgprocessor = MsgProcessor(nbits=config.nbits, hidden_size=config.seanet.dimension)
    msgprocessor = msgprocessor.to(device=device, dtype=dtype)

    return AudioSealWM(encoder=encoder, decoder=decoder, msg_processor=msgprocessor)


def create_detector(
    config: AudioSealDetectorConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> AudioSealDetector:
    detector_config = {**config.seanet, **config.detector}  # type: ignore
    detector = AudioSealDetector(nbits=config.nbits, **detector_config)
    detector = detector.to(device=device, dtype=dtype)
    return detector

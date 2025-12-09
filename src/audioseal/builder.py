# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf
from torch import device, dtype
from typing_extensions import TypeAlias

from audioseal.models import (
    AudioSealDetector,
    AudioSealWM,
    MsgProcessor,
    NormalizationProcessor,
)

# We use different SEANet implementations based on Python version.
# For 3.10 and above: Moshi's SEANetEncoder and SEANetDecoder.
# For 3.9 and below: Audiocraft's SEANetEncoder and SEANetDecoder..
if sys.version_info >= (3, 10):
    from audioseal.libs.moshi.modules.seanet import (
        SEANetDecoder,
        SEANetEncoder,
        SEANetEncoderKeepDimension,
    )
else:
    from audioseal.libs.audiocraft.modules.seanet import (
        SEANetDecoder,
        SEANetEncoder,
        SEANetEncoderKeepDimension,
    )


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
    output_dim: int = 32


@dataclass
class AudioSealWMConfig:
    nbits: int
    seanet: SEANetConfig
    decoder: DecoderConfig
    normalizer: bool = False


@dataclass
class AudioSealDetectorConfig:
    nbits: int
    seanet: SEANetConfig
    detector: DetectorConfig = field(default_factory=lambda: DetectorConfig())
    normalizer: bool = False


def as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    elif isinstance(obj, DictConfig):
        return OmegaConf.to_container(obj)  # type: ignore
    else:
        raise NotImplementedError(f"Unsupported type for config: {type(obj)}")


def create_generator(
    config: AudioSealWMConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> AudioSealWM:
    """Create a generator from hparams"""

    #  Currently the encoder hparams are the same as
    # SEANet, but this can be changed in the future.
    seanet_config = config.seanet

    encoder_config = as_dict(seanet_config)
    decoder_config = {**as_dict(config.seanet), **as_dict(config.decoder)}

    encoder = SEANetEncoder(**encoder_config)
    encoder = encoder.to(device=device, dtype=dtype)

    decoder = SEANetDecoder(**decoder_config)
    decoder = decoder.to(device=device, dtype=dtype)

    msgprocessor = MsgProcessor(
        nbits=config.nbits, hidden_size=seanet_config.dimension)
    msgprocessor = msgprocessor.to(device=device, dtype=dtype)

    normalizer = NormalizationProcessor() if getattr(
        config, "normalizer", False) else None

    return AudioSealWM(encoder=encoder, decoder=decoder, msg_processor=msgprocessor, normalizer=normalizer)


def create_detector(
    config: AudioSealDetectorConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> AudioSealDetector:

    _detector = {"output_dim": 32}
    detector_config = {**as_dict(config.seanet), **_detector}

    encoder = SEANetEncoderKeepDimension(**detector_config)
    normalizer = NormalizationProcessor() if getattr(
        config, "normalizer", False) else None

    detector = AudioSealDetector(
        encoder=encoder, normalizer=normalizer, nbits=config.nbits)
    detector = detector.to(device=device, dtype=dtype)
    return detector

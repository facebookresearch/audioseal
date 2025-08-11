# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Watermarking and detection for speech audios

A Pytorch-based localized algorithm for proactive detection
of the watermarkings in AI-generated audios, with very fast
detector.

"""

__version__ = "0.1.8"


from audioseal import builder
from audioseal.loader import AudioSeal
from audioseal.models import AudioSealDetector, AudioSealWM, MsgProcessor

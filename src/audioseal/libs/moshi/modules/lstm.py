# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from dataclasses import dataclass

import torch
from torch import nn

from audioseal.libs.moshi.modules.streaming import StreamingModule


@torch.jit.script
@dataclass
class _StreamingLSTMState:
    previous_h: tp.Optional[torch.Tensor] = None
    previous_c: tp.Optional[torch.Tensor] = None
    streamable: bool = False

    def init_stream(self) -> None:
        self.streamable = True

    def is_streamable(self) -> bool:
        return self.streamable

    def reset(self):
        self.previous_h = None
        self.previous_c = None


class RawStreamingLSTM(StreamingModule[_StreamingLSTMState]):
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    @torch.jit.export
    def _init_streaming_state(self, batch_size: int) -> _StreamingLSTMState:
        return _StreamingLSTMState()

    def _empty_state(self) -> _StreamingLSTMState:
        return _StreamingLSTMState()

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(2, 0, 1)
        if not self._streaming_state.is_streamable():
            y, (h, _) = self.lstm(x)
        else:
            previous_h = self._streaming_state.previous_h
            previous_c = self._streaming_state.previous_c
            if previous_h is None or previous_c is None:
                y, (h, c) = self.lstm(x)
            else:
                y, (h, c) = self.lstm(x, (previous_h, previous_c))

            self._streaming_state.previous_h = h
            self._streaming_state.previous_c = c
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y

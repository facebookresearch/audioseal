# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple

import torch

from audiocraft.modules.seanet import SEANetEncoder


class MsgProcessor(torch.nn.Module):
    """
    Apply the secret message to the encoder output.
    Args:
        nbits: Number of bits used to generate the message. Must be non-zero
        hidden_size: Dimension of the encoder output
    """

    def __init__(self, nbits: int, hidden_size: int):
        super().__init__()
        assert nbits > 0, "MsgProcessor should not be built in 0bit watermarking"
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.msg_processor = torch.nn.Embedding(2 * nbits, hidden_size)

    def forward(self, hidden: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        """
        Build the embedding map: 2 x k -> k xh, then sum on the first dim
        msg is a binary tensor of size b x k
        """
        # create indices to take from embedding layer
        indices = 2 * torch.arange(msg.shape[-1]).to(msg.device)  # k: 0 2 4 ... 2k
        indices = indices.repeat(msg.shape[0], 1)  # b x k
        indices = (indices + msg).long()
        msg_aux = self.msg_processor(indices)  # bx k -> b x k x h
        msg_aux = msg_aux.sum(dim=-2)  # b x k x h -> b x h
        msg_aux = msg_aux.unsqueeze(-1).repeat(
            1, 1, hidden.shape[2]
        )  # b x h -> b x h x t/f
        hidden = hidden + msg_aux  # -> b x h x t/f
        return hidden


class AudioSealWM(torch.nn.Module):
    """
    Generate watermarking for a given audio signal
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        msg_processor: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # The build should take care of validating the dimensions between component
        self.msg_processor = msg_processor
        self._mesage: Optional[torch.Tensor] = None

    @property
    def message(self) -> Optional[torch.Tensor]:
        return self._mesage

    @message.setter
    def message(self, message: torch.Tensor) -> None:
        self._message = message

    def get_watermark(
        self,
        x: torch.Tensor,
        message: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get the watermark from an audio tensor and a message.
        If the input message is None, a random message of
        n bits {0,1} will be generated
        """
        hidden = self.encoder(x)

        if self.msg_processor is not None:
            if message is None:
                message = self.message or torch.randint(
                    0, 2, (x.shape[0], self.msg_processor.nbits), device=x.device
                )

            hidden = self.msg_processor(hidden, message)
        return self.decoder(hidden)[
            ..., : x.size(-1)
        ]  # trim output cf encodec codebase

    def forward(
        self,
        x: torch.Tensor,
        message: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Apply the watermarking to the audio signal x with a tune-down ratio (default 1.0)"""
        wm = self.get_watermark(x, message)
        return x + alpha * wm


class SEANetEncoderKeepDimension(SEANetEncoder):
    """
    similar architecture to the audiocraft.SEANet encoder but with an extra step that
    projects the output dimension to the same input dimension by repeating
    the sequential

    Args:
        SEANetEncoder (_type_): _description_
        output_dim (int): Output dimension
    """

    def __init__(self, *args, output_dim=8, **kwargs):

        self.output_dim = output_dim
        super().__init__(*args, **kwargs)
        # Adding a reverse convolution layer
        self.reverse_convolution = torch.nn.ConvTranspose1d(
            in_channels=self.dimension,
            out_channels=self.output_dim,
            kernel_size=math.prod(self.ratios),
            stride=math.prod(self.ratios),
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_nframes = x.shape[-1]
        x = self.model(x)
        x = self.reverse_convolution(x)
        # make sure dim didn't change
        return x[:, :, :orig_nframes]


class AudioSealDetector(torch.nn.Module):
    """
    Detect the watermarking from an audio signal
    Args:
        SEANetEncoderKeepDimension (_type_): _description_
        nbits (int): The number of bits in the secret message. The watermarks (if detected)
            will have size 2 + nbits, where the first two items indicate the possibilities
            of a true watermarking (positive / negative scores), he rest is used to decode
            the secret message. In 0bit watermarking (no secret message), the detector just
            returns 2 values.
    """

    def __init__(self, *args, nbits: int = 0, **kwargs):
        super().__init__()
        encoder = SEANetEncoderKeepDimension(*args, **kwargs)
        last_layer = torch.nn.Conv1d(encoder.output_dim, 2 + nbits, 1)
        self.detector = torch.nn.Sequential(encoder, last_layer)

    def decode_message(self, result: torch.Tensor): ...

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        result = self.detector(x)  # b x 2+nbits
        # hardcode softmax on 2 first units used for detection
        result[:, :2, :] = torch.softmax(result[:, :2, :], dim=1)

        # TODO: Return the result and the message as a tuple
        return result[:, :2, :], result[:, 2:, :]

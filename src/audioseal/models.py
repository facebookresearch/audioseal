# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Tuple, Union

import julius
import torch

from audioseal.libs.audiocraft.modules.seanet import SEANetEncoderKeepDimension

logger = logging.getLogger("Audioseal")

COMPATIBLE_WARNING = """
AudioSeal is designed to work at a sample rate 16khz.
Implicit sampling rate usage is deprecated and will be removed in future version.
To remove this warning please add this argument to the function call:
sample_rate = your_sample_rate
"""


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
        Build the embedding map: 2 x k -> k x h, then sum on the first dim
        Args:
            hidden: The encoder output, size: batch x hidden x frames
            msg: The secret message, size: batch x k
        """
        # create indices to take from embedding layer
        indices = 2 * torch.arange(msg.shape[-1]).to(msg.device)  # k: 0 2 4 ... 2k
        indices = indices.repeat(msg.shape[0], 1)  # b x k
        indices = (indices + msg).long()
        msg_aux = self.msg_processor(indices)  # b x k -> b x k x h
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
        self._message: Optional[torch.Tensor] = None

    @property
    def message(self) -> Optional[torch.Tensor]:
        return self._message

    @message.setter
    def message(self, message: torch.Tensor) -> None:
        self._message = message

    def get_watermark(
        self,
        x: torch.Tensor,
        sample_rate: Optional[int] = None,
        message: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get the watermark from an audio tensor and a message.
        If the input message is None, a random message of
        n bits {0,1} will be generated.
        Args:
            x: Audio signal, size: batch x frames
            sample_rate: The sample rate of the input audio (default 16khz as
                currently supported by the main AudioSeal model)
            message: An optional binary message, size: batch x k
        """
        length = x.size(-1)
        if sample_rate is None:
            logger.warning(COMPATIBLE_WARNING)
            sample_rate = 16_000
        assert sample_rate
        if sample_rate != 16000:
            x = julius.resample_frac(x, old_sr=sample_rate, new_sr=16000)
        hidden = self.encoder(x)

        if self.msg_processor is not None:
            if message is None:
                if self.message is None:
                    message = torch.randint(
                        0, 2, (x.shape[0], self.msg_processor.nbits), device=x.device
                    )
                else:
                    message = self.message.to(device=x.device)
            else:
                if message.ndim == 1:
                    message = message.unsqueeze(0).repeat(x.shape[0], 1)
                message = message.to(device=x.device)  # type: ignore

            hidden = self.msg_processor(hidden, message)

        watermark = self.decoder(hidden)

        if sample_rate != 16000:
            watermark = julius.resample_frac(
                watermark, old_sr=16000, new_sr=sample_rate
            )

        return watermark[..., :length]  # trim output cf encodec codebase

    def forward(
        self,
        x: torch.Tensor,
        sample_rate: Optional[int] = None,
        message: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Apply the watermarking to the audio signal x with a tune-down ratio (default 1.0)"""
        if sample_rate is None:
            logger.warning(COMPATIBLE_WARNING)
            sample_rate = 16_000
        wm = self.get_watermark(x, sample_rate=sample_rate, message=message)
        return x + alpha * wm


class AudioSealDetector(torch.nn.Module):
    """
    Detect the watermarking from an audio signal
    Args:
        SEANetEncoderKeepDimension (_type_): _description_
        nbits (int): The number of bits in the secret message. The result will have size
            of 2 + nbits, where the first two items indicate the possibilities of the
            audio being watermarked (positive / negative scores), he rest is used to decode
            the secret message. In 0bit watermarking (no secret message), the detector just
            returns 2 values.
    """

    def __init__(self, *args, nbits: int = 0, **kwargs):
        super().__init__()
        encoder = SEANetEncoderKeepDimension(*args, **kwargs)
        last_layer = torch.nn.Conv1d(encoder.output_dim, 2 + nbits, 1)
        self.detector = torch.nn.Sequential(encoder, last_layer)
        self.nbits = nbits

    def detect_watermark(
        self,
        x: torch.Tensor,
        sample_rate: Optional[int] = None,
        message_threshold: float = 0.5,
        detection_threshold: float = 0.5,
    ) -> Union[Tuple[float, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        A convenience function that returns a probability of an audio being watermarked,
        together with its message in n-bits (binary) format. If the audio is not watermarked,
        the message will be random.
        Args:
            x: Audio signal, size: batch x frames
            sample_rate: The sample rate of the input audio
            message_threshold: threshold used to convert the watermark output (probability
                of each bits being 0 or 1) into the binary n-bit message.
        """
        if sample_rate is None:
            logger.warning(COMPATIBLE_WARNING)
            sample_rate = 16_000
        result, message = self.forward(x, sample_rate=sample_rate)  # b x 2+nbits
        detect_prob = (
            torch.count_nonzero(torch.gt(result[:, 1, :], detection_threshold), dim=-1)
            / result.shape[-1]
        )
        if x.shape[0] == 1:
            detect_prob = detect_prob.detach().cpu().item()  # type: ignore
        message = torch.gt(message, message_threshold).int()
        return detect_prob, message

    def decode_message(self, result: torch.Tensor) -> torch.Tensor:
        """
        Decode the message from the watermark result (batch x nbits x frames)
        Args:
            result: watermark result (batch x nbits x frames)
        Returns:
            The message of size batch x nbits, indicating probability of 1 for each bit
        """
        assert (result.dim() > 2 and result.shape[1] == self.nbits) or (
            self.dim() == 2 and result.shape[0] == self.nbits
        ), f"Expect message of size [,{self.nbits}, frames] (get {result.size()})"
        decoded_message = result.mean(dim=-1)
        return torch.sigmoid(decoded_message)

    def forward(
        self,
        x: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect the watermarks from the audio signal
        Args:
            x: Audio signal, size batch x frames
            sample_rate: The sample rate of the input audio
        """
        if sample_rate is None:
            logger.warning(COMPATIBLE_WARNING)
            sample_rate = 16_000
        assert sample_rate
        if sample_rate != 16000:
            x = julius.resample_frac(x, old_sr=sample_rate, new_sr=16000)
        result = self.detector(x)  # b x 2+nbits
        # hardcode softmax on 2 first units used for detection
        result[:, :2, :] = torch.softmax(result[:, :2, :], dim=1)
        message = self.decode_message(result[:, 2:, :])
        return result[:, :2, :], message

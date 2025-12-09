# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import sys
from contextlib import contextmanager
from typing import Optional, Tuple

import torch

if sys.version_info >= (3, 10):
    from audioseal.libs.moshi.modules.seanet import SEANetEncoderKeepDimension
else:
    from audioseal.libs.audiocraft.modules.seanet import SEANetEncoderKeepDimension


logger = logging.getLogger("Audioseal")


@functools.lru_cache(10)
def warn_once(msg: str) -> None:
    """Give logs in limited number of times to avoid flooding stderr."""
    logger.warning(msg)


SAMPLE_RATE_WARN = (
    "Deprecated Warning: `sample_rate` is specified but it will be ignored. \n"
    "Consider removing `sample_rate` in the model call as this is a no-op\n"
    "Starting from AudioSeal 0.2+, audio is not resampled internally to"
    " 16kHz or some predefined sample rates. The user is responsible for"
    " providing the correct sample rate to the model.\n"
)


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
        # k: 0 2 4 ... 2k
        indices = 2 * torch.arange(msg.shape[-1]).to(hidden.device)
        indices = indices.repeat(msg.shape[0], 1)  # b x k
        indices = (indices + msg).long()
        msg_aux = self.msg_processor(indices)  # b x k -> b x k x h
        msg_aux = msg_aux.sum(dim=-2)  # b x k x h -> b x h
        msg_aux = msg_aux.unsqueeze(-1).repeat(
            1, 1, hidden.shape[2]
        )  # b x h -> b x h x t/f
        hidden = hidden + msg_aux  # -> b x h x t/f
        return hidden


class NormalizationProcessor:
    """
    A class for normalizing audio signals, ensuring they fit within a specified envelope
    and achieving consistent loudness levels.

    Attributes:
        window_size (int): The size of the window for processing the signal.
        reference_rms (float): The reference RMS value for loudness normalization.
    """

    def __init__(self, window_size: int = 5, reference_rms: float = 0.1):
        """
        Initializes the NormalizationProcessor with the given window size and reference RMS value.

        Args:
            window_size (int): The size of the window for processing the signal.
            reference_rms (float): The reference RMS value for loudness normalization.
        """
        self.window_size = window_size
        self.reference_rms = reference_rms

    @torch.jit.export
    def compute_rms(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Computes the root mean square (RMS) of the given signal.

        Args:
            signal (torch.Tensor): The input signal tensor of shape (batch, channels, timesteps).

        Returns:
            torch.Tensor: The RMS value of the signal of shape (batch, channels, 1).
        """
        return torch.sqrt(
            torch.mean(signal**2, dim=-1, keepdim=True) + 1e-8
        )  # Adding epsilon for numerical stability

    def fit_inside_envelope(
        self, wav1: torch.Tensor, wav2: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalizes wav2 to fit inside the envelope defined by wav1.

        Args:
            wav1 (torch.Tensor): The reference signal tensor of shape (batch, channels, timesteps).
            wav2 (torch.Tensor): The signal tensor to be normalized of shape (batch, channels, timesteps).

        Returns:
            torch.Tensor: The normalized wav2 tensor of shape (batch, channels, timesteps).
        """
        wav1 = wav1.clone()
        wav2 = wav2.clone()
        # batch size, number of channels, number of samples
        bsz, channel, samples = wav1.shape

        # Create a Hann window for smooth transitions
        hann_window = torch.hann_window(self.window_size, periodic=False).to(
            wav1.device
        )
        normalized_wav2 = torch.zeros_like(wav2)

        overlap = self.window_size // 2
        num_windows = (samples - self.window_size + overlap) // overlap

        # Unfold the signals into overlapping windows
        # shape: (batch, channels, num_windows, window_size)
        unfolded_wav1 = wav1.unfold(-1, self.window_size, overlap)
        # shape: (batch, channels, num_windows, window_size)
        unfolded_wav2 = wav2.unfold(-1, self.window_size, overlap)

        # Compute RMS for each window
        rms_wav1 = torch.sqrt(torch.mean(
            unfolded_wav1**2, dim=-1, keepdim=True))
        rms_wav2 = torch.sqrt(torch.mean(
            unfolded_wav2**2, dim=-1, keepdim=True))

        # Calculate the gain needed to fit wav2 inside wav1's envelope
        gain = rms_wav1 / (rms_wav2 + 1e-8)
        gain = torch.clamp(gain, min=1e-2, max=1.0)

        hann_window_portion = hann_window.view(1, 1, -1)
        normalized_segment = unfolded_wav2 * gain
        normalized_segment *= hann_window_portion

        # Reconstruct the signal from the normalized windows
        normalized_segment = normalized_segment.swapaxes(-1, -2)
        fold = torch.nn.Fold((1, normalized_wav2.shape[-1]), kernel_size=(1, self.window_size), stride=(1, overlap))
        for i_batch in range(bsz):
            for i_channel in range(channel):
                normalized_wav2[i_batch, i_channel, :] = fold(normalized_segment[i_batch, i_channel, :, :].squeeze(0))

        # Handle the last segment
        remaining_samples = samples - num_windows * overlap
        if remaining_samples > 0:
            start = num_windows * overlap
            window_wav1 = wav1[:, :, start:samples]
            window_wav2 = wav2[:, :, start:samples]
            rms_wav1 = self.compute_rms(window_wav1)
            rms_wav2 = self.compute_rms(window_wav2)
            gain = rms_wav1 / (rms_wav2 + 1e-8)
            gain = torch.clamp(gain, min=1e-2, max=1.0)
            hann_window_portion = hann_window[:remaining_samples]
            normalized_segment = window_wav2 * gain
            normalized_segment *= hann_window_portion.unsqueeze(0).unsqueeze(0)
            normalized_wav2[:, :, start:samples] += normalized_segment

        return normalized_wav2

    @torch.jit.export
    def loudness_normalization(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the loudness of the given audio signal to match the reference RMS.

        Args:
            wav (torch.Tensor): The input signal tensor of shape (batch, channels, timesteps).

        Returns:
            torch.Tensor: The loudness-normalized signal tensor of shape (batch, channels, timesteps).
        """
        wav = wav.clone()
        # batch size, number of channels, number of samples
        bsz, channel, samples = wav.shape

        # Create a Hann window for smooth transitions
        hann_window = torch.hann_window(
            self.window_size, periodic=False).to(wav.device)
        normalized_wav = torch.zeros_like(wav)

        overlap = self.window_size // 2
        num_windows = (samples - self.window_size + overlap) // overlap

        # Unfold the signal into overlapping windows
        # shape: (batch, channels, num_windows, window_size)
        unfolded_wav = wav.unfold(-1, self.window_size, overlap)
        rms_wav = torch.sqrt(torch.mean(unfolded_wav**2, dim=-1, keepdim=True))

        # Calculate the gain needed to achieve the reference RMS
        gain = self.reference_rms / (rms_wav + 1e-8)
        gain = torch.clamp(gain, min=1, max=10.0)

        hann_window_portion = hann_window.view(1, 1, -1)
        normalized_segment = unfolded_wav * gain
        normalized_segment *= hann_window_portion

        # Reconstruct the signal from the normalized windows
        normalized_segment = normalized_segment.swapaxes(-1, -2)
        fold = torch.nn.Fold((1, normalized_wav.shape[-1]), kernel_size=(1, self.window_size), stride=(1, overlap))
        for i_batch in range(bsz):
            for i_channel in range(channel):
                normalized_wav[i_batch, i_channel, :] = fold(normalized_segment[i_batch, i_channel, :, :].squeeze(0))


        remaining_samples = samples - num_windows * overlap

        if remaining_samples > 0:
            start = num_windows * overlap
            window_wav = wav[:, :, start:samples]
            rms_wav = self.compute_rms(window_wav)
            gain = self.reference_rms / (rms_wav + 1e-8)
            gain = torch.clamp(gain, min=1, max=10.0)
            hann_window_portion = hann_window[:remaining_samples]
            normalized_segment = window_wav * gain
            normalized_segment *= hann_window_portion.unsqueeze(0).unsqueeze(0)
            normalized_wav[:, :, start:samples] += normalized_segment

        return normalized_wav


class AudioSealWM(torch.nn.Module):
    """
    Generate watermarking for a given audio signal
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        msg_processor: Optional[torch.nn.Module] = None,
        normalizer: Optional[NormalizationProcessor] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # The build should take care of validating the dimensions between component
        self.msg_processor = msg_processor
        self.message = torch.zeros(0)
        self.normalizer = normalizer

    def __prepare_scriptable__(self):
        for _, module in self.named_modules():
            for hook in module._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(module)
        return self

    @torch.jit.export
    def random_message(self, bsz: int):
        if self.msg_processor is not None:
            nbits: int = self.msg_processor.nbits  # type: ignore
        else:
            nbits = 16
        return torch.randint(0, 2, (bsz, nbits))  # type: ignore

    @torch.jit.export
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

        if sample_rate is not None and sample_rate != 16000:
            if not torch.jit.is_scripting():
                warn_once(SAMPLE_RATE_WARN)

        length = x.size(-1)
        hidden = self.encoder(x)

        if self.msg_processor is not None:
            if message is None:
                if self.message.numel() == 0:
                    self.message = self.random_message(x.shape[0])
                message = self.message.to(device=x.device)

            elif message.ndim == 1:
                message = message.unsqueeze(0).repeat(x.shape[0], 1)

            hidden = self.msg_processor(hidden, message)

        # trim padding induced by seanet
        watermark = self.decoder(hidden)[..., :length]

        # fit under envelope. This only works in eager mode
        # as torch.jit.script does not support the Hand window transformation
        if self.normalizer is not None and not torch.jit.is_scripting():
            watermark = self.normalizer.fit_inside_envelope(x, watermark)

        return watermark

    @torch.jit.export
    def forward(
        self,
        x: torch.Tensor,
        sample_rate: Optional[int] = None,
        message: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
    ) -> torch.Tensor:    
        """Apply the watermarking to the audio signal x with a tune-down ratio (default 1.0)"""

        wm = self.get_watermark(x, sample_rate=sample_rate, message=message)

        return x + alpha * wm

    @contextmanager
    def streaming(self, batch_size: int):
        """wrapper of the self.encoder.streaming() context manager for streaming mode"""

        if not hasattr(self.encoder, "streaming"):
            raise NotImplementedError(
                "Streaming not supported: This checkpoint does not support streaming watermarking, "
                "or you install a version of AudioSeal (<0.2) or Python (<3.10) without streaming support, "
                "Please upgrade to the latest version of AudioSeal and Python 3.10+ to use this feature."
            )
        with self.encoder.streaming(batch_size=batch_size):  # type: ignore
            yield



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

    def __init__(
        self,
        encoder: SEANetEncoderKeepDimension,
        normalizer: Optional[NormalizationProcessor] = None,
        nbits: int = 0,
    ):
        super().__init__()
        last_layer = torch.nn.Conv1d(encoder.output_dim, 2 + nbits, 1)
        self.detector = torch.nn.Sequential(encoder, last_layer)
        self.normalizer = normalizer
        self.nbits = nbits

    def __prepare_scriptable__(self):
        for _, module in self.named_modules():
            for hook in module._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(module)
        return self

    @torch.jit.export
    def detect_watermark(
        self,
        x: torch.Tensor,
        sample_rate: Optional[int] = None,
        message_threshold: float = 0.5,
        detection_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A convenience function that returns a probability of an audio being watermarked,
        together with its message in n-bits (binary) format. If the audio is not watermarked,
        the message will be random.
        Args:
            x: Audio signal, size: batch x frames
            sample_rate: The sample rate of the input audio
            message_threshold: threshold used to convert the watermark output (probability
                of each bits being 0 or 1) into the binary n-bit message.
            detection_threshold: threshold to convert the softmax output to binary indicating
                the probability of the audio being watermarked
        Returns:
            detect_prob: A float indicating the probability of the audio being watermarked
            message: A binary tensor of size batch x nbits, indicating the probability of each bit being 1
        """
        result, message = self.forward(x, sample_rate=sample_rate)  # b x 2+nbits
        detect_prob = (
            torch.count_nonzero(
                torch.gt(result[:, 1, :], detection_threshold), dim=-1) / result.shape[-1]
        )
        message = torch.gt(message, message_threshold).int()
        return detect_prob, message

    @torch.jit.export
    def decode_message(self, result: torch.Tensor) -> torch.Tensor:
        """
        Decode the message from the watermark result (batch x nbits x frames)
        Args:
            result: watermark result (batch x nbits x frames)
        Returns:
            The message of size batch x nbits, indicating probability of 1 for each bit
        """
        decoded_message = result.mean(dim=-1)
        return torch.sigmoid(decoded_message)

    @torch.jit.export
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
        if self.normalizer is not None and not torch.jit.is_scripting():
            x = self.normalizer.loudness_normalization(x)

        if sample_rate is not None and sample_rate != 16000:
            if not torch.jit.is_scripting():
                warn_once(SAMPLE_RATE_WARN)

        result = self.detector(x)  # b x 2+nbits
        # hardcode softmax on 2 first units used for detection
        result[:, :2, :] = torch.softmax(result[:, :2, :], dim=1)
        message = self.decode_message(result[:, 2:, :])
        return result[:, :2, :], message

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Example attacks using different audio effects. 
# For full list of atacks, check 
# https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/utils/audio_effects.py
#
#
import typing as tp

import julius
import torch


def generate_pink_noise(length: int) -> torch.Tensor:
    """
    Generate pink noise using Voss-McCartney algorithm with PyTorch.
    """
    num_rows = 16
    array = torch.randn(num_rows, length // num_rows + 1)
    reshaped_array = torch.cumsum(array, dim=1)
    reshaped_array = reshaped_array.reshape(-1)
    reshaped_array = reshaped_array[:length]
    # Normalize
    pink_noise = reshaped_array / torch.max(torch.abs(reshaped_array))
    return pink_noise


def audio_effect_return(
    tensor: torch.Tensor, mask: tp.Optional[torch.Tensor]
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the mask if it was in the input otherwise only the output tensor"""
    if mask is None:
        return tensor
    else:
        return tensor, mask


class AudioEffects:
    @staticmethod
    def speed(
        tensor: torch.Tensor,
        speed_range: tuple = (0.5, 1.5),
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Function to change the speed of a batch of audio data.
        The output will have a different length !

        Parameters:
        audio_batch (torch.Tensor): The batch of audio data in torch tensor format.
        speed (float): The speed to change the audio to.

        Returns:
        torch.Tensor: The batch of audio data with the speed changed.
        """
        speed = torch.FloatTensor(1).uniform_(*speed_range)
        new_sr = int(sample_rate * 1 / speed)
        resampled_tensor = julius.resample_frac(tensor, sample_rate, new_sr)
        if mask is None:
            return resampled_tensor
        else:
            return resampled_tensor, torch.nn.functional.interpolate(
                mask, size=resampled_tensor.size(-1), mode="nearest-exact"
            )

    @staticmethod
    def updownresample(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        intermediate_freq: int = 32000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        orig_shape = tensor.shape
        # upsample
        tensor = julius.resample_frac(tensor, sample_rate, intermediate_freq)
        # downsample
        tensor = julius.resample_frac(tensor, intermediate_freq, sample_rate)

        assert tensor.shape == orig_shape
        return audio_effect_return(tensor=tensor, mask=mask)

    @staticmethod
    def echo(
        tensor: torch.Tensor,
        volume_range: tuple = (0.1, 0.5),
        duration_range: tuple = (0.1, 0.5),
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Attenuating the audio volume by a factor of 0.4, delaying it by 100ms,
        and then overlaying it with the original.

        :param tensor: 3D Tensor representing the audio signal [bsz, channels, frames]
        :param echo_volume: volume of the echo signal
        :param sample_rate: Sample rate of the audio signal.
        :return: Audio signal with reverb.
        """

        # Create a simple impulse response
        # Duration of the impulse response in seconds
        duration = torch.FloatTensor(1).uniform_(*duration_range)
        volume = torch.FloatTensor(1).uniform_(*volume_range)

        n_samples = int(sample_rate * duration)
        impulse_response = torch.zeros(n_samples).type(tensor.type()).to(tensor.device)

        # Define a few reflections with decreasing amplitude
        impulse_response[0] = 1.0  # Direct sound

        impulse_response[int(sample_rate * duration) - 1] = (
            volume  # First reflection after 100ms
        )

        # Add batch and channel dimensions to the impulse response
        impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)

        # Convolve the audio signal with the impulse response
        reverbed_signal = julius.fft_conv1d(tensor, impulse_response)

        # Normalize to the original amplitude range for stability
        reverbed_signal = (
            reverbed_signal
            / torch.max(torch.abs(reverbed_signal))
            * torch.max(torch.abs(tensor))
        )

        # Ensure tensor size is not changed
        tmp = torch.zeros_like(tensor)
        tmp[..., : reverbed_signal.shape[-1]] = reverbed_signal
        reverbed_signal = tmp

        return audio_effect_return(tensor=reverbed_signal, mask=mask)

    @staticmethod
    def random_noise(
        waveform: torch.Tensor,
        noise_std: float = 0.001,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Add Gaussian noise to the waveform."""
        noise = torch.randn_like(waveform) * noise_std
        noisy_waveform = waveform + noise
        return audio_effect_return(tensor=noisy_waveform, mask=mask)

    @staticmethod
    def pink_noise(
        waveform: torch.Tensor,
        noise_std: float = 0.01,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Add pink background noise to the waveform."""
        noise = generate_pink_noise(waveform.shape[-1]) * noise_std
        noise = noise.to(waveform.device)
        # Assuming waveform is of shape (bsz, channels, length)
        noisy_waveform = waveform + noise.unsqueeze(0).unsqueeze(0).to(waveform.device)
        return audio_effect_return(tensor=noisy_waveform, mask=mask)

    @staticmethod
    def lowpass_filter(
        waveform: torch.Tensor,
        cutoff_freq: float = 5000,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        return audio_effect_return(
            tensor=julius.lowpass_filter(waveform, cutoff=cutoff_freq / sample_rate),
            mask=mask,
        )

    @staticmethod
    def highpass_filter(
        waveform: torch.Tensor,
        cutoff_freq: float = 500,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        return audio_effect_return(
            tensor=julius.highpass_filter(waveform, cutoff=cutoff_freq / sample_rate),
            mask=mask,
        )

    @staticmethod
    def bandpass_filter(
        waveform: torch.Tensor,
        cutoff_freq_low: float = 300,
        cutoff_freq_high: float = 8000,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Apply a bandpass filter to the waveform by cascading
        a high-pass filter followed by a low-pass filter.

        Parameters:
        - waveform (torch.Tensor): Input audio waveform.
        - low_cutoff (float): Lower cutoff frequency.
        - high_cutoff (float): Higher cutoff frequency.
        - sample_rate (int): The sample rate of the waveform.

        Returns:
        - torch.Tensor: Filtered audio waveform.
        """

        return audio_effect_return(
            tensor=julius.bandpass_filter(
                waveform,
                cutoff_low=cutoff_freq_low / sample_rate,
                cutoff_high=cutoff_freq_high / sample_rate,
            ),
            mask=mask,
        )

    @staticmethod
    def smooth(
        tensor: torch.Tensor,
        window_size_range: tuple = (2, 10),
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Smooths the input tensor (audio signal) using a moving average filter with the given window size.

        Parameters:
        - tensor (torch.Tensor): Input audio tensor. Assumes tensor shape is (batch_size, channels, time).
        - window_size (int): Size of the moving average window.

        Returns:
        - torch.Tensor: Smoothed audio tensor.
        """

        window_size = int(torch.FloatTensor(1).uniform_(*window_size_range))
        # Create a uniform smoothing kernel
        kernel = torch.ones(1, 1, window_size).type(tensor.type()) / window_size
        kernel = kernel.to(tensor.device)

        smoothed = julius.fft_conv1d(tensor, kernel)
        # Ensure tensor size is not changed
        tmp = torch.zeros_like(tensor)
        tmp[..., : smoothed.shape[-1]] = smoothed
        smoothed = tmp

        return audio_effect_return(tensor=smoothed, mask=mask)

    @staticmethod
    def boost_audio(
        tensor: torch.Tensor,
        amount: float = 20,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return audio_effect_return(tensor=tensor * (1 + amount / 100), mask=mask)

    @staticmethod
    def duck_audio(
        tensor: torch.Tensor,
        amount: float = 20,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return audio_effect_return(tensor=tensor * (1 - amount / 100), mask=mask)

    @staticmethod
    def identity(
        tensor: torch.Tensor, mask: tp.Optional[torch.Tensor] = None
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return audio_effect_return(tensor=tensor, mask=mask)

    @staticmethod
    def shush(
        tensor: torch.Tensor,
        fraction: float = 0.001,
        mask: tp.Optional[torch.Tensor] = None
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Sets a specified chronological fraction of indices of the input tensor (audio signal) to 0.

        Parameters:
        - tensor (torch.Tensor): Input audio tensor. Assumes tensor shape is (batch_size, channels, time).
        - fraction (float): Fraction of indices to be set to 0 (from the start of the tensor) (default: 0.001, i.e, 0.1%)

        Returns:
        - torch.Tensor: Transformed audio tensor.
        """
        time = tensor.size(-1)
        shush_tensor = tensor.detach().clone()
        
        # Set the first `fraction*time` indices of the waveform to 0
        shush_tensor[:, :, :int(fraction*time)] = 0.0
                
        return audio_effect_return(tensor=shush_tensor, mask=mask)

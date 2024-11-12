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
import torchaudio

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

    @staticmethod
    def pitch_shift(
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        n_steps: float = 2,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Change the pitch of an audio signal by a given number of steps.
        """
        pitch_shifted = torchaudio.transforms.PitchShift(sample_rate, n_steps=n_steps)(waveform)
        return audio_effect_return(tensor=pitch_shifted, mask=mask)

    @staticmethod
    def reverse(
        tensor: torch.Tensor,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Reverse the audio signal.

        Parameters:
        - tensor (torch.Tensor): Input audio tensor, assuming shape (batch_size, channels, time).
        - mask (torch.Tensor): Optional mask tensor.

        Returns:
        - torch.Tensor: Reversed audio tensor.
        """
        reversed_tensor = torch.flip(tensor, dims=[-1])
        return audio_effect_return(tensor=reversed_tensor, mask=mask)
    
    @staticmethod
    def pitch_shift(
        tensor: torch.Tensor,
        n_steps: float = 2.0,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Change the pitch of the audio signal by a given number of steps.

        Parameters:
        - tensor (torch.Tensor): Input audio tensor, assuming shape (batch_size, channels, time).
        - n_steps (float): Number of pitch steps to shift (positive for higher pitch, negative for lower pitch).
        - sample_rate (int): Sample rate of the audio signal.
        - mask (torch.Tensor): Optional mask tensor.

        Returns:
        - torch.Tensor: Pitch-shifted audio tensor.
        """
        shifted_tensor = julius.pitch_shift(tensor, sample_rate, n_steps)
        return audio_effect_return(tensor=shifted_tensor, mask=mask)
    
    @staticmethod
    def clipping(
        tensor: torch.Tensor,
        clip_value: float = 0.5,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Clip the audio signal to a specific threshold value, distorting the signal.

        Parameters:
        - tensor (torch.Tensor): Input audio tensor, assuming shape (batch_size, channels, time).
        - clip_value (float): Threshold for clipping the audio signal.
        - mask (torch.Tensor): Optional mask tensor.

        Returns:
        - torch.Tensor: Clipped audio tensor.
        """
        clipped_tensor = torch.clamp(tensor, min=-clip_value, max=clip_value)
        return audio_effect_return(tensor=clipped_tensor, mask=mask)


    @staticmethod
    def time_stretch(
        tensor: torch.Tensor,
        stretch_factor: float = 1.2,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Stretch the audio signal in time without changing its pitch.

        Parameters:
        - tensor (torch.Tensor): Input audio tensor, assuming shape (batch_size, channels, time).
        - stretch_factor (float): Factor by which to stretch the audio (greater than 1 for slower, less than 1 for faster).
        - mask (torch.Tensor): Optional mask tensor.

        Returns:
        - torch.Tensor: Time-stretched audio tensor.
        """
        stretched_tensor = julius.time_stretch(tensor, stretch_factor)
        return audio_effect_return(tensor=stretched_tensor, mask=mask)
    
    @staticmethod
    def tremolo(
        tensor: torch.Tensor,
        frequency: float = 5.0,
        depth: float = 0.5,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Apply a tremolo effect to the audio signal by modulating its amplitude.

        Parameters:
        - tensor (torch.Tensor): Input audio tensor, assuming shape (batch_size, channels, time).
        - frequency (float): Frequency of the tremolo effect in Hz.
        - depth (float): Depth of modulation (between 0 and 1).
        - sample_rate (int): Sample rate of the audio signal.
        - mask (torch.Tensor): Optional mask tensor.

        Returns:
        - torch.Tensor: Audio tensor with tremolo effect applied.
        """
        time = torch.arange(tensor.shape[-1], device=tensor.device) / sample_rate
        modulation = (1.0 + depth * torch.sin(2 * torch.pi * frequency * time)) / 2.0
        tremolo_tensor = tensor * modulation.unsqueeze(0).unsqueeze(0)
        return audio_effect_return(tensor=tremolo_tensor, mask=mask)

    @staticmethod
    def flanger(
        tensor: torch.Tensor,
        delay: float = 0.002,
        depth: float = 0.002,
        rate: float = 0.25,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Apply a flanger effect to the audio signal by mixing a delayed version of the signal with itself.

        Parameters:
        - tensor (torch.Tensor): Input audio tensor, assuming shape (batch_size, channels, time).
        - delay (float): Base delay time in seconds.
        - depth (float): Depth of the delay modulation.
        - rate (float): Rate of modulation in Hz.
        - sample_rate (int): Sample rate of the audio signal.
        - mask (torch.Tensor): Optional mask tensor.

        Returns:
        - torch.Tensor: Audio tensor with flanger effect applied.
        """
        time = torch.arange(tensor.shape[-1], device=tensor.device) / sample_rate
        lfo = torch.sin(2 * torch.pi * rate * time) * depth + delay
        lfo_samples = (lfo * sample_rate).long().clamp(0, tensor.shape[-1] - 1)
        delayed_signal = tensor[..., lfo_samples]
        flanger_tensor = tensor + delayed_signal
        return audio_effect_return(tensor=flanger_tensor, mask=mask)
    
    @staticmethod
    def distortion(
        tensor: torch.Tensor,
        gain: float = 20.0,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Apply a distortion effect by amplifying the audio signal and then clipping it.

        Parameters:
        - tensor (torch.Tensor): Input audio tensor, assuming shape (batch_size, channels, time).
        - gain (float): Gain factor for amplifying the signal.
        - mask (torch.Tensor): Optional mask tensor.

        Returns:
        - torch.Tensor: Distorted audio tensor.
        """
        amplified = tensor * gain
        distorted = torch.tanh(amplified)
        return audio_effect_return(tensor=distorted, mask=mask)

    @staticmethod
    def bit_crusher(
        tensor: torch.Tensor,
        bit_depth: int = 8,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Apply a bit crusher effect by reducing the bit depth of the audio signal.

        Parameters:
        - tensor (torch.Tensor): Input audio tensor, assuming shape (batch_size, channels, time).
        - bit_depth (int): Bit depth to reduce to (e.g., 8 bits).
        - mask (torch.Tensor): Optional mask tensor.

        Returns:
        - torch.Tensor: Audio tensor with reduced bit depth.
        """
        scale = 2 ** bit_depth
        crushed_tensor = torch.round(tensor * scale) / scale
        return audio_effect_return(tensor=crushed_tensor, mask=mask)

    @staticmethod
    def vocoder(
        tensor: torch.Tensor,
        modulation_frequency: float = 100.0,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Apply a vocoder effect by modulating the audio signal with a carrier frequency.

        Parameters:
        - tensor (torch.Tensor): Input audio tensor, assuming shape (batch_size, channels, time).
        - modulation_frequency (float): Frequency of the modulation in Hz.
        - sample_rate (int): Sample rate of the audio signal.
        - mask (torch.Tensor): Optional mask tensor.

        Returns:
        - torch.Tensor: Vocoded audio tensor.
        """
        time = torch.arange(tensor.shape[-1], device=tensor.device) / sample_rate
        carrier = torch.sin(2 * torch.pi * modulation_frequency * time)
        vocoded_tensor = tensor * carrier.unsqueeze(0).unsqueeze(0)
        return audio_effect_return(tensor=vocoded_tensor, mask=mask)

    @staticmethod
    def ring_modulation(
        tensor: torch.Tensor,
        modulation_frequency: float = 30.0,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Apply a ring modulation effect to the audio signal, creating a metallic sound.

        Parameters:
        - tensor (torch.Tensor): Input audio tensor, assuming shape (batch_size, channels, time).
        - modulation_frequency (float): Frequency of the modulation in Hz.
        - sample_rate (int): Sample rate of the audio signal.
        - mask (torch.Tensor): Optional mask tensor.

        Returns:
        - torch.Tensor: Ring-modulated audio tensor.
        """
        time = torch.arange(tensor.shape[-1], device=tensor.device) / sample_rate
        modulation = torch.sin(2 * torch.pi * modulation_frequency * time)
        ring_modulated_tensor = tensor * modulation.unsqueeze(0).unsqueeze(0)
        return audio_effect_return(tensor=ring_modulated_tensor, mask=mask)

    @staticmethod
    def granulate(
        tensor: torch.Tensor,
        grain_size: int = 512,
        overlap: float = 0.5,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Apply a granulation effect by breaking the audio into small overlapping grains.

        Parameters:
        - tensor (torch.Tensor): Input audio tensor, assuming shape (batch_size, channels, time).
        - grain_size (int): Size of each grain in samples.
        - overlap (float): Overlap ratio between grains (0 to 1).
        - mask (torch.Tensor): Optional mask tensor.

        Returns:
        - torch.Tensor: Granulated audio tensor.
        """
        step_size = int(grain_size * (1 - overlap))
        grains = [tensor[..., i:i+grain_size] for i in range(0, tensor.shape[-1] - grain_size, step_size)]
        granulated_tensor = torch.cat(grains, dim=-1)
        return audio_effect_return(tensor=granulated_tensor, mask=mask)

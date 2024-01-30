# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Accompanying code for the notebook.
# We need to install matplotlib and jupyter notebook beforehand

import IPython.display as ipd
import matplotlib.pyplot as plt
import torch


def plot_waveform_and_specgram(waveform, sample_rate, title):
    waveform = waveform.squeeze().detach().cpu().numpy()

    num_frames = waveform.shape[-1]
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(time_axis, waveform, linewidth=1)
    ax1.grid(True)
    ax2.specgram(waveform, Fs=sample_rate)

    figure.suptitle(f"{title} - Waveform and specgram")
    plt.show()


def play_audio(waveform, sample_rate):
    if waveform.dim() > 2:
        waveform = waveform.squeeze(0)
    waveform = waveform.detach().cpu().numpy()

    num_channels, *_ = waveform.shape
    if num_channels == 1:
        ipd.display(ipd.Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        ipd.display(ipd.Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")

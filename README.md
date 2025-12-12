# :loud_sound: AudioSeal: Efficient Localized Audio Watermarking

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

This repo contains the official implementation  for **AudioSeal**, a method for efficient audio watermarking, with state-of-the-art robustness and detector speed.

# :rocket: Quick Links:

[[`arXiv`](https://arxiv.org/abs/2401.17264)]
[[🤗`Hugging Face`](https://huggingface.co/facebook/audioseal)]
[[`Colab Notebook`](https://colab.research.google.com/github/facebookresearch/audioseal/blob/master/examples/colab.ipynb)]
[[`Webpage`](https://pierrefdz.github.io/publications/audioseal/)]
[[`Blog`](https://about.fb.com/news/2024/06/releasing-new-ai-research-models-to-accelerate-innovation-at-scale/)]
[[`Press`](https://www.technologyreview.com/2024/06/18/1094009/meta-has-created-a-way-to-watermark-ai-generated-speech/)]

![fig](https://github.com/facebookresearch/audioseal/assets/1453243/5d8cd96f-47b5-4c34-a3fa-7af386ed59f2)

# :sparkles: Key Updates:

- 2024-12-12: AudioSeal 0.2 is out, with streaming support and other improvement
- 2024-06-17: Training code is now available. Check the [instruction](./docs/TRAINING.md)!!!
- 2024-05-31: Our paper gets accepted at ICML'24 :)
- 2024-04-02: We have updated our license to full MIT license (including the license for the model weights) ! Now you can use AudioSeal in commercial application too!
- 2024-02-29: AudioSeal 0.1.2 is out, with more bug fixes for resampled audios and updated notebooks


# :book: Abstract

**AudioSeal** introduces a novel audio watermarking using **ocalized watermarking** and a novel perceptual loss. It jointly trains two components: a **generator** that embeds an imperceptible watermark into audio and a **detector** that identifies watermark fragments in long or edited audio files.

- **Key Features:**
  - **Localized watermarking** at the sample level (1/16,000 of a second). AudioSeal works well with other sampling rates as well (24 khZ, 44.5 kHz, 48 kHz)
  - Minimal impact on audio quality.
  - **Robust** against various audio edits like compression, re-encoding, and noise addition.
  - **Fast, single-pass detection** designed to surpass existing models significantly in speed — achieving detection up to **two orders of magnitude faster**, making it ideal for large-scale and real-time applications.


# :mate: Installation

### Requirements:
- Python >= 3.8 (>= 3.10 for streaming support)
- Pytorch >= 1.13.0
- [Omegaconf](https://omegaconf.readthedocs.io/)
- [Numpy](https://pypi.org/project/numpy/)
- [einops](https://github.com/arogozhnikov/einops) (for streaming support)

### Install from PyPI:
```
pip install audioseal
```

To install from source: Clone this repo and install in editable mode:

```
git clone https://github.com/facebookresearch/audioseal
cd audioseal
pip install -e .
```

# :gear: Models

You can find all the model checkpoints on the [Hugging Face Hub](https://huggingface.co/facebook/audioseal). We provide the checkpoints for the following models:

- [AudioSeal Generator](src/audioseal/cards/audioseal_wm_16bits.yaml):
  Takes an audio signal (as a waveform) and outputs a watermark of the same size as the input, which can be added to the input to watermark it. Optionally, it can also take a secret 16-bit message to embed in the watermark.
- [AudioSeal Detector](src/audioseal/cards/audioseal_detector_16bits.yaml):
  Takes an audio signal (as a waveform) and outputs the probability that the input contains a watermark at each sample (every 1/16k second). Optionally, it may also output the secret message encoded in the watermark.

Note that the message is optional and has no influence on the detection output. It may be used to identify a model version for instance (up to $2**16=65536$ possible choices).

# :abacus: Usage

Here’s a quick example of how you can use AudioSeal’s API to embed and detect watermarks:

```python

from audioseal import AudioSeal

# model name corresponds to the YAML card file name found in audioseal/cards
model = AudioSeal.load_generator("audioseal_wm_16bits")
model.eval()

# Other way is to load directly from the checkpoint
# model =  Watermarker.from_pretrained(checkpoint_path, device = wav.device)

# a torch tensor of shape (batch, channels, samples) and a sample rate
# It is important to process the audio to the same sample rate as the model
# expects. The default AudioSeal should work well with 16kHz and 24kHz, and 
# in the case of 48 khZ, it should work well for most speech audios
wav = [load audio wav into a tensor of BatchxChannelxTime]

watermark = model.get_watermark(wav)

# Optional: you can add a 16-bit message to embed in the watermark
# msg = torch.randint(0, 2, (wav.shape(0), model.msg_processor.nbits), device=wav.device)
# watermark = model.get_watermark(wav, message = msg)

watermarked_audio = wav + watermark

detector = AudioSeal.load_detector("audioseal_detector_16bits")

# To detect the messages in the high-level.
result, message = detector.detect_watermark(watermarked_audio)

print(result) # result is a float number indicating the probability of the audio being watermarked,
print(message)  # message is a binary vector of 16 bits


# To detect the messages in the low-level.
result, message = detector(watermarked_audio)

# result is a tensor of size batch x 2 x frames, indicating the probability (positive and negative) of watermarking for each frame
# A watermarked audio should have result[:, 1, :] > 0.5
print(result[:, 1 , :])  

# Message is a tensor of size batch x 16, indicating of the probability of each bit to be 1.
# message will be a random tensor if the detector detects no watermarking from the audio
print(message)  
```

# :abacus: Streaming support

Starting AudioSeal 0.2, you can run the watermarking over the stream of audio signals. The API is `model.streaming(batch_size),
which will enable the convolutional cache during the watermark generation. Ensure to put this within context, so the cache is
safely cleaned after the session:

```python

model = AudioSeal.load_generator("audioseal_wm_streaming")
model.eval()

audio = [audio chunks]
streaming_watermarked_audio = []

with model.streaming(batch_size=1):
    
    # Watermark each incoming chunk of the streaming audio
    for chunk in audio:
        watermarked_chunk = model(chunk, sample_rate=sr, message=secret_mesage, alpha=1)
        streaming_watermarked_audio.append(watermarked_chunk)
  
streaming_watermarked_audio = torch.cat(streaming_watermarked_audio, dim=1)


# You can detect a chunk of watermarked output, or the whole audio:

detector = AudioSeal.load_detector("audioseal_detector_streaming")
detector.eval()

wm_chunk = 100
partial_result, _ = detector.detect_watermark(streaming_watermarked_audio[:, :, :wm_chunk])


full_result, _ = detector.detect_watermark(streaming_watermarked_audio)

```
See [example notebook](examples/Getting_started.ipynb) for full details.


# :brain: Train your own watermarking model

See [here](./docs/TRAINING.md) for details on how to train your own Watermarking model.


# See Also

The team also develops other open-source watermarking solutions:
- [WMAR](https://github.com/facebookresearch/wmar): Autoregressive watermarking models for images
- [Video Seal](https://github.com/facebookresearch/videoseal): Open and efficient video watermarking
- [WAM](https://github.com/facebookresearch/watermark-anything): Watermark Any Images with Localization



# 🎮 Troubleshooting

- If you encounter the error `ValueError: not enough values to unpack (expected 3, got 2)`, this is because we expect a batch of audio  tensors as inputs. Add one
dummy batch dimension to your input (e.g. `wav.unsqueeze(0)`, see [example notebook for getting started](examples/Getting_started.ipynb)).

- In Windows machines, if you encounter the error `KeyError raised while resolving interpolation: "Environmen variable 'USER' not found"`: This is due to an old checkpoint
uploaded to the model hub, which is not compatible in Windows. Try to invalidate the cache by removing the files in `C:\Users\<USER>\.cache\audioseal`
and re-run again.

- If you use torchaudio to handle your audios and encounter the error `Couldn't find appropriate backend to handle uri ...`, this is due to newer version of
torchaudio does not handle the default backend well. Either downgrade your torchaudio to `2.1.0` or earlier, or install `soundfile` as your audio backend.

# :heart: Acknowledgements

We borrow the code with some adaptations from the following repos:
- [AudioCraft](https://github.com/facebookresearch/audiocraft/) in `libs/audiocraft/`.
- [Moshi](https://github.com/kyutai-labs/moshi/) in `libs/moshi/`.


# :handshake: Contributions

 We welcome Pull Requests with improvements or suggestions.
 If you want to flag an issue or propose an improvement, but don't know how to realize it, create a GitHub Issue.


# 🧾 License

- The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).


# ✍️ Citation

If you find this repository useful, please consider giving a star :star: and please cite as:

```
@article{sanroman2024proactive,
  title={Proactive Detection of Voice Cloning with Localized Watermarking},
  author={San Roman, Robin and Fernandez, Pierre and Elsahar, Hady and D´efossez, Alexandre and Furon, Teddy and Tran, Tuan},
  journal={ICML},
  year={2024}
}
```

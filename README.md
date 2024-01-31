# :loud_sound: AudioSeal: Proactive Localized Watermarking

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

We introduce AudioSeal, a method for speech localized watermarking
, with state-of-the-art detector speed without compromising the watermarking robustness. It jointly trains a generator that embeds a watermark in the audio, and a detector that detects the watermarked fragments in longer audios, even in the presence of editing.
Audioseal achieves state-of-the-art detection performance of both natural and synthetic speech at the sample level (1/16k second resolution), it generates limited alteration of signal quality and is robust to many types of audio editing. 
Audioseal is designed with a fast, single-pass detector, that significantly surpasses existing models in speed — achieving detection up to two orders of magnitude faster, making it ideal for large-scale and real-time applications. 

More details can be found in the [paper](https://arxiv.org/pdf/2401.17264.pdf)


![fig](https://github.com/facebookresearch/audioseal/assets/1453243/5d8cd96f-47b5-4c34-a3fa-7af386ed59f2)


# :mate: Installation

AudioSeal requires Python >=3.8, Pytorch >= 1.13.0, [omegaconf](https://omegaconf.readthedocs.io/), [julius](https://pypi.org/project/julius/), and numpy. To install from PyPI:

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

We provide the checkpoints for the following models:

- [AudioSeal Generator](https://dl.fbaipublicfiles.com/audioseal/audioseal_wm_16bits.pth).
  It takes as input an audio signal (as a waveform), and outputs a watermark of the same size as the input, that can be added to the input to watermark it.
  Optionally, it can also take as input a secret message of 16-bits that will be encoded in the watermark.
- [AudioSeal Detector](https://dl.fbaipublicfiles.com/audioseal/audioseal_detector_16bits.pth).
  It takes as input an audio signal (as a waveform), and outputs a probability that the input contains a watermark at each sample of the audio (every 1/16k s).
  Optionally, it may also output the secret message encoded in the watermark.

Note that the message is optional and has no influence on the detection output. It may be used to identify a model version for instance (up to $2**16=65536$ possible choices).

**Note**: We are working to release the training code for anyone wants to build their own watermarker. Stay tuned !

# :abacus: Usage

Audioseal provides a simple API to watermark and detect the watermarks from an audio sample. Example usage:

```python

from audioseal import AudioSeal

# model name corresponds to the YAML card file name found in audioseal/cards
model = AudioSeal.load_generator("audioseal_wm_16bits")

# Other way is to load directly from the checkpoint
# model =  Watermarker.from_pretrained(checkpoint_path, device = wav.device)

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

# result is a tensor of size batch x 2 x frames, indicating the probablity (positive and negative) of watermarking for each frame
# A watermarked audio should have result[:, 1, :] > 0.5
print(result[:, 1 , :])  

# Message is a tensor of size batch x 16, indicating of the probability of each bit to be 1.
# message will be a random tensor if the detector detects no watermarking from the audio
print(message)  
```

<!-- # Want to contribute?

 We welcome [Pull Requests](https://github.com/fairinternal/fair-getting-started-recipe/pulls) with improvements or suggestions.
 If you want to flag an issue or propose an improvement, but dont' know how to realize it, create a [GitHub Issue](https://github.com/fairinternal/fair-getting-started-recipe/issues).


# Thanks to:
* Jack Urbaneck, Matthew Muckley, Pierre Gleize,  Ashutosh Kumar, Megan Richards, Haider Al-Tahan, and Vivien Cabannes for contributions and feedback
* The CIFAR10 [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
) on which the training is based
* [Hydra Lightning Template](https://github.com/ashleve/lightning-hydra-template) for inspiration on code organization -->

# License

- The code in this repository is released under the MIT license as found in the [LICENSE file](LICENSE).
- The models weights in this repository are released under the CC-BY-NC 4.0 license as found in the [LICENSE_weights file](LICENSE_weights).

# Maintainers:
- [Tuan Tran](https://github.com/antoine-tran)
- [Hady Elsahar](https://github.com/hadyelsahar)
- [Pierre Fernandez](https://github.com/pierrefdz)
- [Robin San Roman](https://github.com/Sparker17)

# Citation

If you find this repository useful, please consider giving a star :star: and please cite as:

```
@article{sanroman2024proactive,
  title={Proactive Detection of Voice Cloning with Localized Watermarking},
  author={San Roman, Robin and Fernandez, Pierre and Elsahar, Hady and D´efossez, Alexandre and Furon, Teddy and Tran, Tuan},
  journal={arXiv preprint},
  year={2024}
}
```

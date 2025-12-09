# Changelog

All notable changes to AudioSeal are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [0.2.0] - 2025-12-09

- Add new models with streaming support (`audioseal_wm_streaming`, `audioseal_detector_streaming`)
- Refactor code to make the functions torchscriptable
- Deprecate the internal resampling during watermark generation and detection
- Fix bugs in load user-defined secret messages with mis-matching batch size
- Update example notebooks

## [0.1.5 - 0.1.8] - 2025-04-29

- Fix bugs in loading model in new PyTorch (2.6+)
- Add support for loading from other HF spaces

## [0.1.4] - 2024-06-24

- Update scripts to new training code
- Fix bugs in loading custom fine-tuned model (https://github.com/facebookresearch/audioseal/issues/37)

## [0.1.3] - 2024-04-30

- Fix bug in getting the watermark with non-empty message created in CPU, while the model is loaded in CUDA
- Update Fix bug in building the model card programmatically (not via .YAML file using OmegaConf)
- Add support for HuggingFace Hub, now we can load the model from HF. Unit tests are updated

## [0.1.2] - 2024-02-29

- Add py.typed to make audioseal mypy-friendly
- Add the option to resample the input audio's sample rate to the expected sample rate of the model (https://github.com/facebookresearch/audioseal/pull/18)
- Move `attacks.py` to non-core code base of audioseal
- Remove duplicate module `SEANetEncoderKeepDimension` in `audioseal.lib.audiocraft.modules.seanet` and `audioseal.models`

## [0.1.1] - 2024-02-04

- Fix [issue](https://github.com/facebookresearch/audioseal/issues/7) in installing audioseal from pypi due to conflict with audiocraft package
- Fix typos in example notebooks
- Update checkpoint to be Windows-compatible

## [0.1.0] - 2024-02-01

- Initial release

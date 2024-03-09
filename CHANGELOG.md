# Changelog

All notable changes to AudioSeal are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

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

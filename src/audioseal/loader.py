# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import pickle
import re
from dataclasses import fields
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, TypeVar, Union, cast
from urllib.parse import urlparse  # noqa: F401

import torch
from omegaconf import DictConfig, OmegaConf

import audioseal
from audioseal.builder import (
    AudioSealDetectorConfig,
    AudioSealWMConfig,
    DataType,
    Device,
    create_detector,
    create_generator,
)
from audioseal.models import AudioSealDetector, AudioSealWM

AudioSealT = TypeVar("AudioSealT", AudioSealWMConfig, AudioSealDetectorConfig)


class ModelLoadError(RuntimeError):
    """Raised when the model loading fails"""


def _convert_model_state_dict(
    state_dict: Dict[str, Any], key_map: Mapping[str, str]
) -> Dict[str, Any]:
    """Convert a model state dictionary to fairseq2.

    :param state_dict:
        The original model state dictionary.
    :param key_map:
        A map of regex patterns to fairseq2 model keys.

    :returns:
        A converted model state dictionary that is compatible with fairseq2.
    """
    new_state_dict = {}

    def get_new_key(old_key: str) -> str:
        for old_pattern, replacement in key_map.items():
            if (new_key := re.sub(old_pattern, replacement, old_key)) != old_key:
                return new_key

        return old_key

    # Convert module keys from fairseq to fairseq2.
    for old_key in state_dict.keys():
        new_key = get_new_key(old_key)

        new_state_dict[new_key] = state_dict[old_key]

    return new_state_dict


def _get_path_from_env(var_name: str) -> Optional[Path]:
    pathname = os.getenv(var_name)
    if not pathname:
        return None

    try:
        return Path(pathname)
    except ValueError as ex:
        raise RuntimeError(f"Expect valid pathname, get '{pathname}'.") from ex


def _get_cache_dir(env_names: List[str]):
    """Re-use cache dir from a list of existing caches"""
    for env in env_names:
        cache_dir = _get_path_from_env(env)
        if cache_dir:
            break
    else:
        cache_dir = Path("~/.cache").expanduser().resolve()

    # Create a sub-dir to not mess up with existing caches
    cache_dir = cache_dir / "audioseal"
    cache_dir.mkdir(exist_ok=True, parents=True)

    return cache_dir


def _safe_load_checkpoint(
    model_path: Union[str, Path], device: Union[str, torch.device] = "cpu"
):
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    except pickle.UnpicklingError as _:
        # This happens in torch 2.6+ . We make a quick hack to allow omegaconf DictConfig
        # to be passed as a global
        import omegaconf

        torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    return ckpt


def _update_state_dict(model: torch.nn.Module, state_dict: Dict[str, Any]):
    def keymap(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        key_map = {
            r"parametrizations.weight.original0": r"weight_g",
            r"parametrizations.weight.original1": r"weight_v",
        }
        return _convert_model_state_dict(state_dict, key_map)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as _:
        # This happens when loading AudioSeal checkpoint trained on newer torch
        # in an application using older torch version. Make the conversion of
        # the state dict
        state_dict = keymap(state_dict)
        model.load_state_dict(state_dict)


def _load_hf_model_checkpoint(
    model_uri: str,
    cache_dir: Path,
    device: Union[str, torch.device] = "cpu",
):
    # Only load from known HF repos
    hf_repos = ["facebook/audioseal", "meta-adtp/audioseal"]

    for repo in hf_repos:
        if model_uri.startswith(repo):
            hf_filename = model_uri[len(repo) + 1 :]
            try:
                import huggingface_hub as hf_hub

                try:
                    file = hf_hub.hf_hub_download(
                        repo_id=repo,
                        repo_type="model",
                        filename=hf_filename,
                        local_dir=cache_dir,
                        library_name="audioseal",
                        library_version=audioseal.__version__,
                    )
                except Exception as _:

                    # Most likely we access a gated repo, try with token
                    file = hf_hub.hf_hub_download(
                        repo_id=repo,
                        repo_type="model",
                        filename=hf_filename,
                        local_dir=cache_dir,
                        library_name="audioseal",
                        token=True,
                        library_version=audioseal.__version__,
                    )

                return _safe_load_checkpoint(file, device=device)

            except ModuleNotFoundError as ex:
                raise ModelLoadError(
                    f"The model path {model_uri} seems to be a direct HF path, "
                    "but you do not install Huggingface_hub. Install with for example "
                    "`pip install huggingface_hub` to use this feature."
                ) from ex

    return None


def load_model_checkpoint(
    model_path: Union[Path, str],
    device: Union[str, torch.device] = "cpu",
):
    if Path(model_path).is_file():
        return _safe_load_checkpoint(model_path, device=device)

    cache_dir = _get_cache_dir(
        ["AUDIOSEAL_CACHE_DIR", "AUDIOCRAFT_CACHE_DIR", "XDG_CACHE_HOME"]
    )
    parts = urlparse(str(model_path))

    # Load HF model
    model = _load_hf_model_checkpoint(
        str(model_path), cache_dir=cache_dir, device=device
    )
    if model is not None:
        return model

    # Load external public model
    if parts.scheme == "https":
        hash_ = sha1(parts.path.encode()).hexdigest()[:24]
        return torch.hub.load_state_dict_from_url(
            str(model_path), model_dir=cache_dir, map_location=device, file_name=hash_
        )
    else:
        raise ModelLoadError(f"Path or uri {model_path} is unknown or does not exist")


def load_local_model_config(model_card: str) -> Optional[DictConfig]:
    config_file = Path(__file__).parent / "cards" / (model_card + ".yaml")
    if Path(config_file).is_file():
        return cast(DictConfig, OmegaConf.load(config_file.resolve()))
    else:
        return None


class AudioSeal:

    @staticmethod
    def parse_model(
        model_card_or_path: str,
        model_type: Type[AudioSealT],
        nbits: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], AudioSealT]:
        """
        Parse the information from the model card or checkpoint path using
        the schema `model_type` that defines the model type
        """
        # Get the raw checkpoint and config from the local model cards
        config = load_local_model_config(model_card_or_path)

        if config:
            assert "checkpoint" in config, f"Checkpoint missing in {model_card_or_path}"
            config_dict = OmegaConf.to_container(config)
            assert isinstance(
                config_dict, dict
            ), f"Cannot parse config from {model_card_or_path}"
            checkpoint = config_dict.pop("checkpoint")
            checkpoint = load_model_checkpoint(checkpoint)

        # Get the raw checkpoint and config from the checkpoint path
        else:
            config_dict = {}
            checkpoint = load_model_checkpoint(model_card_or_path)

        if "xp.cfg" in checkpoint:
            config_dict = {**checkpoint["xp.cfg"], **config_dict}  # type: ignore

        model_config = AudioSeal.parse_config(config_dict, config_type=model_type, nbits=nbits)  # type: ignore

        if "model" in checkpoint:
            checkpoint = checkpoint["model"]

        return checkpoint, model_config

    @staticmethod
    def parse_config(
        config: Dict[str, Any],
        config_type: Type[AudioSealT],
        nbits: Optional[int] = None,
    ) -> AudioSealT:

        assert "seanet" in config, f"missing seanet backbone config in {config}"

        # Patch 1: Resolve the variables in the checkpoint
        config = OmegaConf.create(config)  # type: ignore
        OmegaConf.resolve(config)  # type: ignore
        config = OmegaConf.to_container(config)  # type: ignore

        # Patch 2: Put decoder, encoder and detector outside seanet
        seanet_config = config["seanet"]
        for key_to_patch in ["encoder", "decoder", "detector"]:
            if key_to_patch in seanet_config:
                config_to_patch = config.get(key_to_patch) or {}
                config[key_to_patch] = {
                    **config_to_patch,
                    **seanet_config.pop(key_to_patch),
                }

        config["seanet"] = seanet_config

        # Patch 3: Put nbits into config if specified
        if nbits and "nbits" not in config:
            config["nbits"] = nbits

        # remove attributes not related to the model_type
        result_config = {}
        assert config, "Empty config"
        for field in fields(config_type):
            if field.name in config:
                result_config[field.name] = config[field.name]

        schema = OmegaConf.structured(config_type)
        schema.merge_with(result_config)
        return schema

    @staticmethod
    def load_generator(
        model_card_or_path: str,
        nbits: Optional[int] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> AudioSealWM:
        """Load the AudioSeal generator from the model card"""
        checkpoint, config = AudioSeal.parse_model(
            model_card_or_path,
            AudioSealWMConfig,
            nbits=nbits,
        )

        model = create_generator(config, device=device, dtype=dtype)
        _update_state_dict(model, checkpoint)
        return model

    @staticmethod
    def load_detector(
        model_card_or_path: str,
        nbits: Optional[int] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> AudioSealDetector:
        checkpoint, config = AudioSeal.parse_model(
            model_card_or_path,
            AudioSealDetectorConfig,
            nbits=nbits,
        )
        model = create_detector(config, device=device, dtype=dtype)
        _update_state_dict(model, checkpoint)
        return model

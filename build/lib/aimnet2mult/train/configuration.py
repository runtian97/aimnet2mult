"""Configuration helpers for mixed-fidelity training."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence, Tuple

from omegaconf import OmegaConf

from .fidelity_specific_utils import validate_fidelity_configuration

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PACKAGE_DIR.parent / "config" / "default_model.yaml"
DEFAULT_TRAIN_CONFIG_PATH = PACKAGE_DIR.parent / "config" / "default_train_mixed_fidelity.yaml"

__all__ = [
    "DEFAULT_MODEL_PATH",
    "DEFAULT_TRAIN_CONFIG_PATH",
    "load_configs",
]


def load_configs(
    extra_config_paths: Sequence[str],
    model_path: str | None,
    cli_overrides: Sequence[str],
) -> Tuple[OmegaConf, OmegaConf]:
    """
    Load model/train configuration files, apply overrides, and validate.
    """
    model_path = model_path or str(DEFAULT_MODEL_PATH)
    train_cfg = OmegaConf.load(str(DEFAULT_TRAIN_CONFIG_PATH))
    model_cfg = OmegaConf.load(model_path)

    logging.info("Using model definition: %s", model_path)
    logging.info("Using default training configuration: %s", DEFAULT_TRAIN_CONFIG_PATH)

    for cfg_path in extra_config_paths:
        logging.info("Merging configuration: %s", cfg_path)
        train_cfg = OmegaConf.merge(train_cfg, OmegaConf.load(cfg_path))

    if cli_overrides:
        logging.info("Applying CLI overrides: %s", ", ".join(cli_overrides))
        overrides_cfg = OmegaConf.from_dotlist(cli_overrides)
        train_cfg = OmegaConf.merge(train_cfg, overrides_cfg)

    _convert_fidelity_keys_to_int(train_cfg)
    _prune_placeholder_entries(train_cfg)
    validate_fidelity_configuration(train_cfg)
    _resolve_relative_paths(train_cfg, _determine_reference_dir(extra_config_paths))

    logging.info("--- Model Configuration ---\n%s", OmegaConf.to_yaml(model_cfg))
    logging.info("--- Training Configuration ---\n%s", OmegaConf.to_yaml(train_cfg))

    return model_cfg, train_cfg


def _determine_reference_dir(extra_config_paths: Sequence[str]) -> str:
    if extra_config_paths:
        return os.path.dirname(os.path.abspath(extra_config_paths[0]))
    return os.getcwd()


def _convert_fidelity_keys_to_int(train_cfg) -> None:
    data_cfg = getattr(train_cfg, "data", None)
    if data_cfg is None:
        return

    if hasattr(data_cfg, "fidelity_datasets") and data_cfg.fidelity_datasets is not None:
        new_dict = {}
        for key, value in data_cfg.fidelity_datasets.items_ex(resolve=False):
            new_dict[int(key)] = value
        data_cfg.fidelity_datasets = new_dict

    if hasattr(data_cfg, "fidelity_weights") and data_cfg.fidelity_weights is not None:
        new_dict = {}
        for key, value in data_cfg.fidelity_weights.items_ex(resolve=False):
            new_dict[int(key)] = value
        data_cfg.fidelity_weights = new_dict

    sae_cfg = getattr(data_cfg, "sae", None)
    if sae_cfg and hasattr(sae_cfg, "energy") and sae_cfg.energy is not None:
        if hasattr(sae_cfg.energy, "files") and sae_cfg.energy.files is not None:
            new_dict = {}
            for key, value in sae_cfg.energy.files.items_ex(resolve=False):
                new_dict[int(key)] = value
            sae_cfg.energy.files = new_dict


def _prune_placeholder_entries(train_cfg) -> None:
    data_cfg = getattr(train_cfg, "data", None)
    if data_cfg is None:
        return

    if hasattr(data_cfg, "fidelity_datasets"):
        to_remove = []
        for fid, path in data_cfg.fidelity_datasets.items_ex(resolve=False):
            if str(path) == "???":
                to_remove.append(fid)
        for fid in to_remove:
            del data_cfg.fidelity_datasets[fid]

    if hasattr(data_cfg, "fidelity_weights") and data_cfg.fidelity_weights is not None:
        to_remove = []
        for fid in list(data_cfg.fidelity_weights.keys()):
            if fid not in data_cfg.fidelity_datasets:
                to_remove.append(fid)
        for fid in to_remove:
            del data_cfg.fidelity_weights[fid]

    sae_cfg = getattr(data_cfg, "sae", None)
    if sae_cfg and hasattr(sae_cfg, "energy") and sae_cfg.energy is not None:
        if hasattr(sae_cfg.energy, "files") and sae_cfg.energy.files is not None:
            to_remove = []
            for fid, path in sae_cfg.energy.files.items_ex(resolve=False):
                if fid not in data_cfg.fidelity_datasets or str(path) == "???":
                    to_remove.append(fid)
            for fid in to_remove:
                del sae_cfg.energy.files[fid]


def _resolve_relative_paths(train_cfg, reference_dir: str) -> None:
    data_cfg = train_cfg.data
    for fid, path in data_cfg.fidelity_datasets.items():
        if not os.path.isabs(path):
            data_cfg.fidelity_datasets[fid] = os.path.abspath(os.path.join(reference_dir, path))

    sae_cfg = getattr(data_cfg, "sae", None)
    if sae_cfg and hasattr(sae_cfg, "energy") and sae_cfg.energy is not None:
        files = getattr(sae_cfg.energy, "files", None)
        if files is not None:
            for fid, sae_path in files.items():
                sae_path = str(sae_path)
                if sae_path and not os.path.isabs(sae_path):
                    files[fid] = os.path.abspath(os.path.join(reference_dir, sae_path))

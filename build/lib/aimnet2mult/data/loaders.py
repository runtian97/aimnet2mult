"""Data loader utilities for mixed-fidelity training."""

from __future__ import annotations

import logging
from typing import Tuple

import yaml

from .collate import mixed_fidelity_collate_fn
from .mixed import MixedFidelityDataset
from .sampler import MixedFidelitySampler

from .sgdataset import SizeGroupedDataset

__all__ = ["create_mixed_fidelity_loaders", "MixedFidelityDataLoader"]


def create_mixed_fidelity_loaders(cfg) -> Tuple["MixedFidelityDataLoader", "MixedFidelityDataLoader"]:
    """
    Build train/validation loaders according to the OmegaConf configuration.
    """
    from ignite import distributed as idist

    fidelity_datasets = {}
    keys = list(cfg.data.x) + list(cfg.data.y)

    if idist.get_world_size() > 1 and not cfg.data.get("ddp_load_full_dataset", False):
        shard = (idist.get_local_rank(), idist.get_world_size())
    else:
        shard = None

    for fid_level, fid_path in cfg.data.fidelity_datasets.items():
        logging.info("Loading fidelity %s dataset from %s", fid_level, fid_path)
        dataset = SizeGroupedDataset(fid_path, keys=keys, shard=shard)
        logging.info("Fidelity %s: %d samples", fid_level, len(dataset))
        fidelity_datasets[fid_level] = dataset

    fidelity_weights = cfg.data.get("fidelity_weights", None)
    if fidelity_weights is not None:
        fidelity_weights = dict(fidelity_weights)

    fidelity_offset = cfg.get("fidelity_offset", 100)
    mf_dataset_train = MixedFidelityDataset(
        fidelity_datasets,
        fidelity_weights=fidelity_weights,
        fidelity_offset=fidelity_offset,
    )
    mf_dataset_train = _apply_sae(mf_dataset_train, cfg)

    logging.info("Total training samples: %d", len(mf_dataset_train))
    for fid, length in mf_dataset_train.get_fidelity_lengths().items():
        logging.info("  Fidelity %s: %d samples", fid, length)

    if cfg.data.get("separate_val", True):
        mf_dataset_train, mf_dataset_val = mf_dataset_train.random_split(
            1 - cfg.data.val_fraction,
            cfg.data.val_fraction,
        )
        logging.info(
            "Split dataset: train=%d, val=%d",
            len(mf_dataset_train),
            len(mf_dataset_val),
        )
    else:
        mf_dataset_val = mf_dataset_train

    train_sampler = MixedFidelitySampler(dataset=mf_dataset_train, **cfg.data.samplers.train.kwargs)
    val_sampler = MixedFidelitySampler(dataset=mf_dataset_val, **cfg.data.samplers.val.kwargs)

    mf_dataset_train.set_loader_mode(cfg.data.x, cfg.data.y)
    mf_dataset_val.set_loader_mode(cfg.data.x, cfg.data.y)

    train_loader = MixedFidelityDataLoader(
        mf_dataset_train,
        train_sampler,
        **cfg.data.loaders.train,
    )
    val_loader = MixedFidelityDataLoader(
        mf_dataset_val,
        val_sampler,
        **cfg.data.loaders.val,
    )
    return train_loader, val_loader


class MixedFidelityDataLoader:
    """Thin iterable wrapper over the sampler + dataset combination."""

    def __init__(self, dataset, sampler, num_workers: int = 0, pin_memory: bool = True) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def __len__(self) -> int:
        return len(self.sampler)

    def __iter__(self):
        for batch_indices in self.sampler:
            batch_data = self.dataset.get_loader_batch(batch_indices)
            yield mixed_fidelity_collate_fn(
                batch_data,
                x_keys=self.dataset.x_keys,
                y_keys=self.dataset.y_keys,
                template_specs=getattr(self.dataset, "template_specs", None),
            )


def _apply_sae(mf_dataset, cfg):
    if cfg.data.get("sae") is None or "energy" not in cfg.data.sae:
        return mf_dataset

    sae_cfg = cfg.data.sae.energy
    mode = sae_cfg.get("mode", "linreg")

    per_fidelity_sae = {}
    if hasattr(sae_cfg, "files") and sae_cfg.files is not None:
        for fid_str, path in sae_cfg.files.items():
            fid = int(fid_str)
            with open(str(path), "r") as handle:
                per_fidelity_sae[fid] = yaml.safe_load(handle)
    else:
        with open(str(sae_cfg.file), "r") as handle:
            shared_sae = yaml.safe_load(handle)
        for fid in mf_dataset.fidelity_levels:
            per_fidelity_sae[fid] = shared_sae

    for fid in mf_dataset.fidelity_levels:
        sae = per_fidelity_sae[fid]
        dataset = mf_dataset.get_dataset(fid)
        offset = fid * mf_dataset.fidelity_offset
        shifted_sae = {int(k) + offset: v for k, v in sae.items()}

        if mode == "linreg":
            dataset.apply_peratom_shift("energy", "energy", sap_dict=shifted_sae)
        elif mode == "logratio":
            dataset.apply_pertype_logratio("energy", "energy", sap_dict=shifted_sae)

    return mf_dataset

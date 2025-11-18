"""Samplers for mixed-fidelity training."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .mixed import MixedFidelityDataset

__all__ = ["MixedFidelitySampler"]


class MixedFidelitySampler:
    """
    Create batches that contain molecules from different fidelity levels.

    The sampler groups molecules by size to avoid padding overhead while
    allowing the collate function to simply stack aligned arrays.
    """

    def __init__(
        self,
        dataset: MixedFidelityDataset,
        batch_size: int = 32,
        batch_mode: str = "molecules",
        shuffle: bool = True,
        batches_per_epoch: int = -1,
        sampling_strategy: str = "weighted",
    ) -> None:
        assert batch_mode in {"molecules", "atoms"}, f"Unknown batch_mode {batch_mode}"
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.shuffle = shuffle
        self.batches_per_epoch = batches_per_epoch
        self.sampling_strategy = sampling_strategy

    def __len__(self) -> int:
        if self.batches_per_epoch > 0:
            return self.batches_per_epoch

        total_samples = 0
        for fid in self.dataset.fidelity_levels:
            ds = self.dataset.get_dataset(fid)
            for group in ds.values():
                molecules = len(group["energy"])
                if self.batch_mode == "molecules":
                    total_samples += molecules
                else:
                    total_samples += molecules * group["numbers"].shape[1]

        return max(1, int(np.ceil(total_samples / self.batch_size)))

    def __iter__(self):
        return iter(self._create_batches())

    def _create_batches(self):
        samples = self._build_sample_pool()
        if self.shuffle:
            np.random.shuffle(samples)
        batches = self._create_mixed_batches(samples)

        if self.batches_per_epoch > 0:
            if len(batches) > self.batches_per_epoch:
                batches = batches[: self.batches_per_epoch]
            elif len(batches) < self.batches_per_epoch:
                extra = self.batches_per_epoch - len(batches)
                indices = np.random.choice(len(batches), extra, replace=True)
                batches.extend([batches[i] for i in indices])

        return batches

    def _build_sample_pool(self) -> List[Tuple[int, str, int]]:
        all_samples: List[Tuple[int, str, int]] = []

        for fid in self.dataset.fidelity_levels:
            dataset = self.dataset.get_dataset(fid)
            weight = self.dataset.fidelity_weights.get(fid, 1.0)
            for group_key, group in dataset.items():
                n_molecules = len(group["energy"])
                if n_molecules == 0:
                    continue
                for mol_idx in range(n_molecules):
                    sample = (fid, group_key, mol_idx)
                    if self.sampling_strategy == "weighted":
                        copies = max(1, int(weight * 10))
                        all_samples.extend([sample] * copies)
                    else:
                        all_samples.append(sample)
        return all_samples

    def _create_mixed_batches(self, samples: List[Tuple[int, str, int]]):
        size_groups: dict[int, List[Tuple[int, str, int]]] = {}
        for sample in samples:
            fid, group_key, mol_idx = sample
            dataset = self.dataset.get_dataset(fid)
            group = dataset[group_key]
            n_atoms = group["numbers"].shape[1]
            size_groups.setdefault(n_atoms, []).append(sample)

        batches: List[List[Tuple[int, str, int]]] = []
        if self.batch_mode == "molecules":
            for n_atoms in sorted(size_groups.keys()):
                group_samples = size_groups[n_atoms]
                for idx in range(0, len(group_samples), self.batch_size):
                    batch = group_samples[idx : idx + self.batch_size]
                    if batch:
                        batches.append(batch)
        else:  # atoms
            for n_atoms in sorted(size_groups.keys()):
                group_samples = size_groups[n_atoms]
                current_batch: List[Tuple[int, str, int]] = []
                current_atoms = 0

                for sample in group_samples:
                    if current_atoms + n_atoms > self.batch_size and current_batch:
                        batches.append(current_batch)
                        current_batch = [sample]
                        current_atoms = n_atoms
                    else:
                        current_batch.append(sample)
                        current_atoms += n_atoms

                if current_batch:
                    batches.append(current_batch)

        return batches

"""Mixed-fidelity dataset implementations."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Union

import numpy as np

from .sgdataset import SizeGroupedDataset
from .base import MultiFidelityDataset

__all__ = ["MixedFidelityDataset"]


class MixedFidelityDataset(MultiFidelityDataset):
    """
    Dataset that pre-applies fidelity-dependent offsets to atomic numbers.

    Offsetting atomic numbers ahead of time makes it possible to assemble
    batches that contain samples from multiple fidelities without additional
    padding logic in the collate step.
    """

    def __init__(
        self,
        fidelity_datasets: Dict[int, SizeGroupedDataset],
        fidelity_weights: Dict[int, float] | None = None,
        fidelity_offset: int = 200,
        _skip_offset_application: bool = False,
    ) -> None:
        super().__init__(fidelity_datasets, fidelity_weights)
        self.fidelity_offset = fidelity_offset

        if not _skip_offset_application:
            self._apply_atomic_number_offsets()

        logging.info("Initialized MixedFidelityDataset with offset=%d", fidelity_offset)
        for fid in self.fidelity_levels:
            ds = self.get_dataset(fid)
            sample_numbers = next(iter(ds.values()))["numbers"][0]
            logging.info("  Fidelity %d sample atomic numbers: %s", fid, sample_numbers[:5].tolist())

    def _apply_atomic_number_offsets(self) -> None:
        for fid in self.fidelity_levels:
            offset = fid * self.fidelity_offset
            ds = self.get_dataset(fid)
            logging.info("Applying offset %d to fidelity %d", offset, fid)

            for group in ds.values():
                if "numbers" not in group:
                    continue
                original_numbers = group["numbers"]
                mask = original_numbers > 0
                # Convert to int32 to prevent overflow when adding offset
                shifted = original_numbers.astype('int32')
                shifted[mask] = original_numbers[mask].astype('int32') + offset
                group["numbers"] = shifted

    def __getitem__(self, index: int):
        cumulative = 0
        for fid in sorted(self.fidelity_levels):
            ds = self.get_dataset(fid)
            fid_len = len(ds)
            if index < cumulative + fid_len:
                local_index = index - cumulative
                x, y = ds[local_index]
                return x, y, fid
            cumulative += fid_len
        raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")

    def random_split(self, *fractions: float, seed: int | None = None) -> List["MixedFidelityDataset"]:
        """Override random_split to preserve fidelity_offset and skip re-applying offsets."""
        splits = [
            self.datasets[fid].random_split(*fractions, seed=seed) for fid in self.fidelity_levels
        ]

        result: List[MixedFidelityDataset] = []
        for split_index in range(len(fractions)):
            fid_datasets = {
                fid: splits[idx][split_index] for idx, fid in enumerate(self.fidelity_levels)
            }
            result.append(
                MixedFidelityDataset(
                    fid_datasets,
                    fidelity_weights=self.fidelity_weights,
                    fidelity_offset=self.fidelity_offset,
                    _skip_offset_application=True,  # Offsets already applied
                )
            )
        return result

    def get_loader_batch(self, batch_indices: Tuple[int, int, np.ndarray]):
        """
        Efficiently retrieve a batch using array indexing.

        Args:
            batch_indices: Tuple of (fidelity, group_key, indices_array)
                          where indices_array is a numpy array of molecule indices.

        Returns:
            Tuple of (x_dict, y_dict, fidelities_array) with batched data.
        """
        fid, group_key, indices = batch_indices
        dataset = self.get_dataset(fid)
        group = dataset[group_key]

        # Efficient batch indexing - single NumPy operation per key
        x = {}
        for key in self.x_keys:
            if key in group:
                x[key] = group[key][indices]

        y = {}
        for key in self.y_keys:
            if key in group:
                y[key] = group[key][indices]

        # Create fidelity array for the batch
        fidelities = np.full(len(indices), fid, dtype=np.int64)

        return x, y, fidelities

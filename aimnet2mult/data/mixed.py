"""Mixed-fidelity dataset implementations."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

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
    ) -> None:
        super().__init__(fidelity_datasets, fidelity_weights)
        self.fidelity_offset = fidelity_offset
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
                shifted = original_numbers.copy()
                shifted[mask] = original_numbers[mask] + offset
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

    def get_loader_batch(self, indices: List[Tuple[int, str, int]]):
        batch = []
        for fid, group_key, mol_idx in indices:
            dataset = self.get_dataset(fid)
            group = dataset[group_key]
            x = {}
            y = {}
            for key in self.x_keys:
                if key in group:
                    x[key] = group[key][mol_idx]
            for key in self.y_keys:
                if key in group:
                    y[key] = group[key][mol_idx]
            batch.append((x, y, fid))
        return batch

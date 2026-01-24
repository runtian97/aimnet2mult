"""Samplers for mixed-fidelity training."""

from __future__ import annotations

from typing import List, Tuple, Union
import numpy as np

from .mixed import MixedFidelityDataset

__all__ = ["MixedFidelitySampler"]

# Batch index type: (fidelity, group_key, indices_array)
BatchIndices = Tuple[int, int, np.ndarray]


class MixedFidelitySampler:
    """
    Efficient sampler for mixed-fidelity datasets.

    Uses batch indexing (like aimnet2's SizeGroupedSampler) instead of
    per-molecule tuples for better performance.
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
        self._is_single_fidelity = len(dataset.fidelity_levels) == 1

    def __len__(self) -> int:
        if self.batches_per_epoch > 0:
            return self.batches_per_epoch
        return sum(self._get_num_batches_for_group(fid, g)
                   for fid in self.dataset.fidelity_levels
                   for g in self.dataset.get_dataset(fid).groups)

    def _get_num_batches_for_group(self, fid: int, group) -> int:
        n = len(group)
        if n == 0:
            return 0
        if self.batch_mode == "molecules":
            return int(np.ceil(n / self.batch_size))
        else:  # atoms
            return int(np.ceil(n * group["numbers"].shape[1] / self.batch_size))

    def __iter__(self):
        return iter(self._create_batches())

    def _create_batches(self) -> List[BatchIndices]:
        """Create batches using efficient array-based indexing."""
        batches: List[BatchIndices] = []

        for fid in self.dataset.fidelity_levels:
            dataset = self.dataset.get_dataset(fid)
            weight = self.dataset.fidelity_weights.get(fid, 1.0)

            for group_key, group in dataset.items():
                n = len(group)
                if n == 0:
                    continue

                # Create index array for this group
                idx = np.arange(n)
                if self.shuffle:
                    np.random.shuffle(idx)

                # Apply weighting by duplicating samples if needed
                if self.sampling_strategy == "weighted" and weight > 1.0:
                    copies = max(1, int(weight))
                    idx = np.tile(idx, copies)
                    if self.shuffle:
                        np.random.shuffle(idx)

                # Split into batches
                n_batches = self._get_num_batches_for_group(fid, group)
                if n_batches > 0:
                    for idx_batch in np.array_split(idx, n_batches):
                        if len(idx_batch) > 0:
                            batches.append((fid, group_key, idx_batch))

        # Shuffle batches
        if self.shuffle:
            np.random.shuffle(batches)

        # Handle batches_per_epoch
        if self.batches_per_epoch > 0:
            if len(batches) > self.batches_per_epoch:
                batches = batches[:self.batches_per_epoch]
            elif len(batches) < self.batches_per_epoch:
                # Add random duplicates
                extra = self.batches_per_epoch - len(batches)
                indices = np.random.choice(len(batches), extra, replace=True)
                batches.extend([batches[i] for i in indices])

        return batches

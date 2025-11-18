"""Base dataset abstractions for multi-fidelity AIMNet2 training."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


from .sgdataset import SizeGroupedDataset

__all__ = ["MultiFidelityDataset"]


class MultiFidelityDataset:
    """
    Base container that manages multiple fidelity-specific datasets.

    Each fidelity level corresponds to one :class:`SizeGroupedDataset` coming
    from the original AIMNet2 code base. The class keeps track of sampling
    weights, exposes combined length helpers, and configures loader metadata.
    """

    def __init__(
        self,
        fidelity_datasets: Dict[int, SizeGroupedDataset],
        fidelity_weights: Dict[int, float] | None = None,
    ) -> None:
        self.fidelity_levels = sorted(fidelity_datasets.keys())
        self.num_fidelities = len(self.fidelity_levels)
        self.datasets = fidelity_datasets

        if fidelity_weights is None:
            fidelity_weights = {fid: float(fid + 1) for fid in self.fidelity_levels}

        total_weight = sum(fidelity_weights.values())
        self.fidelity_weights = {k: v / total_weight for k, v in fidelity_weights.items()}

        for dataset in self.datasets.values():
            dataset.loader_mode = False

        self.x_keys: List[str] = []
        self.y_keys: List[str] = []
        self.loader_mode = False
        self.template_specs: Dict[str, dict] = {}

    def get_dataset(self, fidelity_level: int) -> SizeGroupedDataset:
        return self.datasets[fidelity_level]

    def __len__(self) -> int:
        return sum(len(ds) for ds in self.datasets.values())

    def get_fidelity_lengths(self) -> Dict[int, int]:
        return {fid: len(ds) for fid, ds in self.datasets.items()}

    def random_split(self, *fractions: float, seed: int | None = None) -> List["MultiFidelityDataset"]:
        splits = [
            self.datasets[fid].random_split(*fractions, seed=seed) for fid in self.fidelity_levels
        ]

        result: List[MultiFidelityDataset] = []
        for split_index in range(len(fractions)):
            fid_datasets = {
                fid: splits[idx][split_index] for idx, fid in enumerate(self.fidelity_levels)
            }
            result.append(
                type(self)(fid_datasets, fidelity_weights=self.fidelity_weights)
            )
        return result

    def set_loader_mode(self, x: Sequence[str] | None, y: Sequence[str] | None) -> None:
        self.loader_mode = True
        self.x_keys = list(x) if x is not None else []
        self.y_keys = list(y) if y is not None else []

        for dataset in self.datasets.values():
            dataset.loader_mode = True
            dataset.x = x
            dataset.y = y or {}

        keys = self.x_keys + self.y_keys
        self.template_specs = self._infer_template_specs(keys)

    def _infer_template_specs(self, keys: Sequence[str]):
        specs: Dict[str, dict] = {}
        ordered_keys: List[str] = []
        seen = set()

        for key in keys:
            if key not in seen:
                ordered_keys.append(key)
                seen.add(key)

        remaining = set(ordered_keys)
        for fid in self.fidelity_levels:
            dataset = self.get_dataset(fid)
            for group in dataset.values():
                for key in list(remaining):
                    if key not in group:
                        continue

                    sample = group[key][0]
                    arr = np.asarray(sample)
                    numbers = group["numbers"][0] if "numbers" in group else None

                    per_atom = False
                    base_atom_dim = None
                    tail_shape = arr.shape
                    if numbers is not None and arr.ndim > 0:
                        mol_numbers = np.asarray(numbers)
                        if arr.shape[0] == mol_numbers.shape[0]:
                            per_atom = True
                            base_atom_dim = arr.shape[0]
                            tail_shape = arr.shape[1:]

                    specs[key] = {
                        "dtype": arr.dtype if hasattr(arr, "dtype") else np.float32,
                        "per_atom": per_atom,
                        "tail_shape": tail_shape,
                        "base_atom_dim": base_atom_dim,
                    }
                    remaining.remove(key)

                if not remaining:
                    break
            if not remaining:
                break

        return specs

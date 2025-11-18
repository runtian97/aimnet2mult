"""Collate helpers for mixed-fidelity batches."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

__all__ = ["mixed_fidelity_collate_fn"]


def mixed_fidelity_collate_fn(
    batch: List[Tuple[dict, dict, int]],
    x_keys: Sequence[str] | None = None,
    y_keys: Sequence[str] | None = None,
    template_specs: Dict[str, dict] | None = None,
):
    """
    Collate function that keeps molecules size-aligned and preserves fidelity indices.
    """
    if not batch:
        raise ValueError("Batch cannot be empty")

    x_list = [item[0] for item in batch]
    y_list = [item[1] for item in batch]
    fidelities = [item[2] for item in batch]

    atom_counts = []
    for sample in x_list:
        numbers = sample.get("numbers")
        atom_counts.append(None if numbers is None else np.asarray(numbers).shape[0])

    x_batch, _ = _collate_with_padding(
        x_list,
        expected_keys=x_keys,
        create_masks=False,
        template_specs=template_specs,
        atom_counts=atom_counts,
    )
    y_batch, y_masks = _collate_with_padding(
        y_list,
        expected_keys=y_keys,
        create_masks=True,
        template_specs=template_specs,
        atom_counts=atom_counts,
    )

    x_batch = _numpy_dict_to_tensors(x_batch)
    y_batch = _numpy_dict_to_tensors(y_batch)
    if y_masks:
        for key, mask in y_masks.items():
            y_batch[f"{key}_mask"] = torch.from_numpy(mask).float()

    fidelities_tensor = torch.tensor(fidelities, dtype=torch.long)
    return x_batch, y_batch, fidelities_tensor


def _collect_keys(dict_list: Sequence[dict], expected_keys: Sequence[str] | None = None) -> List[str]:
    keys = list(expected_keys) if expected_keys else []
    seen = set(keys)
    for item in dict_list:
        for key in item.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def _collate_with_padding(
    dict_list: Sequence[dict],
    expected_keys: Sequence[str] | None = None,
    create_masks: bool = False,
    template_specs: Dict[str, dict] | None = None,
    atom_counts: Sequence[int | None] | None = None,
):
    if not dict_list:
        return {}, {}

    result: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {} if create_masks else {}
    template_specs = template_specs or {}
    atom_counts = atom_counts or [None] * len(dict_list)
    keys = _collect_keys(dict_list, expected_keys)

    for key in keys:
        ref_shape = None
        ref_dtype = None
        values: List[np.ndarray | None] = []
        availability = []

        for item in dict_list:
            if key in item:
                arr = np.asarray(item[key])
                if ref_shape is None:
                    ref_shape = arr.shape
                    ref_dtype = arr.dtype
                values.append(arr)
                availability.append(1.0)
            else:
                values.append(None)
                availability.append(0.0)
                if ref_shape is not None:
                    zeros = np.zeros(ref_shape, dtype=ref_dtype if ref_dtype is not None else np.float32)
                    values[-1] = zeros

        if ref_shape is None:
            spec = template_specs.get(key)
            if spec is None:
                continue
            dtype = spec.get("dtype", np.float32)
            per_atom = spec.get("per_atom", False)
            tail_shape = tuple(spec.get("tail_shape", ()))
            base_atom_dim = spec.get("base_atom_dim", None)
            zeros_list = []
            for atom_count in atom_counts:
                if per_atom:
                    current_atoms = atom_count if atom_count is not None else base_atom_dim
                    if current_atoms is None:
                        continue
                    shape = (current_atoms,) + tail_shape
                else:
                    shape = tail_shape
                zeros_list.append(np.zeros(shape, dtype=dtype))
            if not zeros_list:
                continue
            stacked = np.stack(zeros_list, axis=0)
            result[key] = stacked
            if create_masks:
                masks[key] = np.zeros((len(dict_list), 1), dtype=np.float32)
            continue

        for idx, val in enumerate(values):
            if val is None:
                values[idx] = np.zeros(ref_shape, dtype=ref_dtype if ref_dtype is not None else np.float32)

        stacked = np.stack(values, axis=0)
        result[key] = stacked

        if create_masks:
            mask = np.array(availability, dtype=np.float32).reshape(len(availability), 1)
            masks[key] = mask

    return result, masks


def _numpy_dict_to_tensors(data: Dict[str, np.ndarray]):
    """
    Convert numpy arrays to tensors.

    Datasets are expected to already be in eV-based units (matching AIMNet2):
    - Energy: eV
    - Forces: eV/Angstrom
    - Coordinates: Angstrom
    """
    tensors: Dict[str, torch.Tensor] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value).float()
            tensors[key] = tensor
        else:
            tensors[key] = value
    return tensors

"""Collate helpers for mixed-fidelity batches."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch

__all__ = ["mixed_fidelity_collate_fn"]


def mixed_fidelity_collate_fn(
    batch_data: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray],
    x_keys: Sequence[str] | None = None,
    y_keys: Sequence[str] | None = None,
    template_specs: Dict[str, dict] | None = None,
):
    """
    Efficient collate function for pre-batched data.

    Args:
        batch_data: Tuple of (x_dict, y_dict, fidelities) from get_loader_batch
                   where x_dict and y_dict already have batch dimension.
        x_keys: Expected x keys (unused, kept for compatibility)
        y_keys: Expected y keys (unused, kept for compatibility)
        template_specs: Template specs (unused, kept for compatibility)

    Returns:
        Tuple of (x_batch, y_batch, fidelities_tensor)
    """
    x_dict, y_dict, fidelities = batch_data

    # Convert numpy arrays to tensors
    x_batch = _numpy_dict_to_tensors(x_dict)
    y_batch = _numpy_dict_to_tensors(y_dict)

    # Convert fidelities to tensor
    fidelities_tensor = torch.from_numpy(fidelities).long()

    return x_batch, y_batch, fidelities_tensor


def _numpy_dict_to_tensors(data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
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
            # Use appropriate dtype
            if value.dtype in (np.int32, np.int64):
                tensor = torch.from_numpy(value).long()
            else:
                tensor = torch.from_numpy(value).float()
            tensors[key] = tensor
        else:
            tensors[key] = value
    return tensors

"""
Utility helpers for eager-mode derivative evaluations.

These functions deliberately avoid TorchScript so that higher-order
autograd information (needed e.g. for Hessians) is preserved.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Dict, Tuple

import torch
import yaml
from omegaconf import OmegaConf

from ..models.mixed_fidelity_aimnet2 import MixedFidelityAIMNet2
from .units import HARTREE_TO_EV


class EagerFidelityModel(torch.nn.Module):
    """Eager counterpart of the TorchScript SAE wrapper without detaching outputs."""

    def __init__(
        self,
        model: torch.nn.Module,
        sae: Dict[int, float],
        fidelity_level: int,
        fidelity_offset: int = 100,
    ) -> None:
        super().__init__()
        self.model = model
        self.fidelity_level = fidelity_level
        self.fidelity_offset = fidelity_offset
        self._numbers_shift = self.fidelity_level * self.fidelity_offset

        max_z = max(sae.keys()) if sae else 118
        tensor_size = max(max_z + 1, 119, (fidelity_level + 1) * fidelity_offset + 118)
        self.register_buffer("sae_tensor", torch.zeros(tensor_size, dtype=torch.float32))
        self.register_buffer("hartree_to_ev", torch.tensor(float(HARTREE_TO_EV), dtype=torch.float32))
        for z, energy in sae.items():
            shifted_z = int(z) + self._numbers_shift
            if shifted_z >= self.sae_tensor.numel():
                raise ValueError(
                    f"SAE entry {z} (shifted to {shifted_z}) exceeds tensor capacity {self.sae_tensor.numel()}"
                )
            self.sae_tensor[shifted_z] = energy

    def _shift_atomic_numbers(self, numbers: torch.Tensor) -> torch.Tensor:
        numbers_long = numbers.to(torch.long)
        if self._numbers_shift == 0:
            return numbers_long
        mask = (numbers_long > 0).to(numbers_long.dtype)
        return numbers_long + mask * self._numbers_shift

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        data = dict(data)
        coord = data['coord']
        numbers = data['numbers']
        numbers_long = numbers.to(torch.long)
        shifted_numbers = self._shift_atomic_numbers(numbers_long)
        charge = data['charge']
        mult = data['mult']

        if '_fidelities' in data:
            fidelities = data['_fidelities'].to(torch.long)
        else:
            fidelities = torch.full(
                (coord.size(0),),
                self.fidelity_level,
                dtype=torch.long,
                device=coord.device,
            )
            data['_fidelities'] = fidelities

        data['numbers'] = shifted_numbers
        pred = self.model(data)
        sae_energy = self.sae_tensor[shifted_numbers].sum(dim=-1)
        energy_hartree = pred['energy'] + sae_energy
        energy_ev = energy_hartree * self.hartree_to_ev

        pred = dict(pred)
        pred['energy'] = energy_ev
        pred['coord'] = coord
        pred['numbers'] = numbers_long
        pred['charge'] = charge
        pred['mult'] = mult
        return pred

__all__ = [
    "load_eager_model",
    "energy_forces_hessian",
    "calculate_hessian_from_forces",
    "DEFAULT_MODEL_PATH",
]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "examples",
    "mixed_fidelity",
    "model.pt",
)


def _resolve_path(*parts: str) -> str:
    return os.path.abspath(os.path.join(PROJECT_ROOT, *parts))


def load_eager_model(
    fidelity: int = 0,
    checkpoint: str | None = None,
    device: str | torch.device = "cpu",
) -> FidelityModelWithSAE:
    """
    Load the trained mixed-fidelity model together with SAE corrections in eager mode.

    Parameters
    ----------
    fidelity:
        Fidelity index whose SAE mapping should be used.
    checkpoint:
        Optional path to the ``model.pt`` checkpoint. Defaults to the example checkpoint.
    device:
        Target device for the eager model.
    """
    device = torch.device(device)
    checkpoint = os.path.abspath(checkpoint or DEFAULT_MODEL_PATH)

    train_cfg_path = _resolve_path("examples", "mixed_fidelity", "train.yaml")
    model_cfg_path = _resolve_path("examples", "mixed_fidelity", "model.yaml")

    train_cfg = OmegaConf.load(train_cfg_path)
    model_cfg = OmegaConf.to_container(
        OmegaConf.load(model_cfg_path),
        resolve=True,
    )
    config_dir = os.path.dirname(train_cfg_path)

    base_model = MixedFidelityAIMNet2(
        base_model_config=model_cfg,
        num_fidelities=len(train_cfg.data.fidelity_datasets),
        fidelity_offset=train_cfg.get("fidelity_offset", 100),
        use_fidelity_readouts=train_cfg.get("use_fidelity_readouts", True),
    ).to(device)
    checkpoint_data = torch.load(checkpoint, map_location=device)
    base_model.load_state_dict(checkpoint_data["model"])
    base_model.eval()

    sae_files = train_cfg.data.sae.energy.files
    if str(fidelity) in sae_files:
        sae_file = sae_files[str(fidelity)]
    else:
        sae_file = sae_files[fidelity]
    sae_path = os.path.abspath(os.path.join(config_dir, sae_file))
    with open(sae_path, "r") as handle:
        sae_mapping = yaml.safe_load(handle)

    wrapper = EagerFidelityModel(
        base_model,
        sae_mapping,
        fidelity_level=fidelity,
        fidelity_offset=train_cfg.get("fidelity_offset", 100),
    ).to(device)
    wrapper.eval()
    return wrapper


def energy_forces_hessian(
    model: torch.nn.Module,
    coord: torch.Tensor,
    numbers: torch.Tensor,
    charge: torch.Tensor,
    mult: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Evaluate energy, forces, and the Hessian for a batch of coordinates.

    Notes
    -----
    - ``coord`` is expected to have shape ``(B, N, 3)``.
    - ``numbers`` should be a long tensor; ``charge`` and ``mult`` are float tensors.
    - Returns detached tensors in a dictionary.
    """
    coord = coord.clone().detach().requires_grad_(True)
    numbers = numbers.to(torch.long)
    charge = charge.to(coord)
    mult = mult.to(coord)

    def energy_sum(c: torch.Tensor) -> torch.Tensor:
        return model(
            {"coord": c, "numbers": numbers, "charge": charge, "mult": mult}
        )["energy"].sum()

    energy = energy_sum(coord)
    forces = -torch.autograd.grad(energy, coord, create_graph=True)[0]

    hessian_full = torch.autograd.functional.hessian(
        energy_sum,
        coord,
        vectorize=False,
    )
    batch, natom, ndim = coord.shape
    hessian_full = hessian_full.view(batch, natom, ndim, batch, natom, ndim)
    batch_indices = torch.arange(batch, device=coord.device)
    hessian = hessian_full[batch_indices, :, :, batch_indices, :, :]

    return {
        "energy": energy.detach(),
        "forces": forces.detach(),
        "hessian": hessian.detach(),
    }


def calculate_hessian_from_forces(
    forces: torch.Tensor,
    coord: torch.Tensor,
) -> torch.Tensor:
    """
    Compute (and sanitise) the Hessian from forces using the user-supplied pattern.

    Parameters
    ----------
    forces:
        Tensor with shape ``(B, N, 3)`` or ``(N, 3)`` representing ``-dE/dR``.
    coord:
        Tensor with shape ``(B, N, 3)`` or ``(B, N + 1, 3)`` (last atom padding).
        Must retain the computation graph from the forward pass.

    Returns
    -------
    torch.Tensor
        Hessian with shape ``(B, N, 3, N, 3)`` with ``NaN``/``Inf`` replaced by zeros.
    """
    if forces.dim() == 2:
        forces = forces.unsqueeze(0)
    if coord.dim() == 2:
        coord = coord.unsqueeze(0)

    batch, natoms, ndim = forces.shape

    if coord.shape[1] == natoms + 1:
        coord = coord[:, :-1]

    flat_forces = forces.reshape(-1)
    grads = []
    for component in flat_forces.unbind():
        grad = torch.autograd.grad(
            component,
            coord,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            grad = torch.zeros_like(coord)
        grads.append(grad)

    hessian_full = -torch.stack(grads).view(batch, natoms, ndim, batch, natoms, ndim)
    batch_idx = torch.arange(batch, device=coord.device)
    hessian = hessian_full[batch_idx, :, :, batch_idx, :, :]
    return torch.nan_to_num(hessian)

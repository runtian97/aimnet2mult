"""
ASE calculators for AIMNet2 models.

Supports both:
- .jpt: TorchScript JIT models (dispersion embedded)
- .pt: aimnet v2 dict format (model_yaml + state_dict, external dispersion)
"""

import warnings
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


def _load_pt_model(path: str, device: torch.device):
    """Load aimnet v2 dict format (.pt) model.

    Returns:
        (model, metadata_dict)
    """
    from .config import build_module

    data = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(data, dict) or 'model_yaml' not in data:
        raise ValueError(f"Not a valid v2 .pt file: missing 'model_yaml' key")

    model_config = yaml.safe_load(data['model_yaml'])
    model = build_module(model_config)

    load_result = model.load_state_dict(data['state_dict'], strict=False)
    # Warn on unexpected mismatches (LRCoulomb keys are expected to be missing/extra)
    real_missing = [k for k in load_result.missing_keys
                    if not k.startswith('outputs.lrcoulomb.') and not k.startswith('outputs.srcoulomb.')]
    real_unexpected = [k for k in load_result.unexpected_keys
                       if not k.startswith('outputs.lrcoulomb.') and not k.startswith('outputs.dftd3.')]
    if real_missing or real_unexpected:
        msg_parts = []
        if real_missing:
            msg_parts.append(f"Missing keys: {real_missing}")
        if real_unexpected:
            msg_parts.append(f"Unexpected keys: {real_unexpected}")
        warnings.warn(f"State dict mismatch: {'; '.join(msg_parts)}")

    model = model.to(device)

    # Preserve float64 precision for atomic shifts (SAE values)
    if hasattr(model, 'outputs') and hasattr(model.outputs, 'atomic_shift'):
        model.outputs.atomic_shift.shifts = model.outputs.atomic_shift.shifts.double()

    metadata = {
        'format_version': data.get('format_version', 2),
        'cutoff': data['cutoff'],
        'needs_coulomb': data.get('needs_coulomb', False),
        'needs_dispersion': data.get('needs_dispersion', False),
        'coulomb_mode': data.get('coulomb_mode', 'none'),
        'd3_params': data.get('d3_params'),
        'has_embedded_lr': data.get('has_embedded_lr', False),
        'implemented_species': data.get('implemented_species', []),
    }

    return model, metadata


class AIMNet2Calculator(Calculator):
    """
    ASE calculator for AIMNet2 models (.jpt and .pt formats).

    - .jpt: TorchScript JIT with NN + SAE + optional embedded D3BJ
    - .pt: aimnet v2 dict with SAE baked into atomic_shift, external dispersion

    Example:
        >>> from aimnet2mult.calculators import AIMNet2Calculator
        >>> calc = AIMNet2Calculator('model_fid0.jpt')
        >>> atoms = molecule('H2O')
        >>> atoms.calc = calc
        >>> energy = atoms.get_potential_energy()
    """

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        model_path: str,
        charge: float = 0.0,
        mult: float = 1.0,
        device: str = 'cpu',
        **kwargs
    ):
        """
        Initialize AIMNet2 calculator.

        Args:
            model_path: Path to model (.jpt or .pt)
            charge: Molecular charge
            mult: Spin multiplicity
            device: Device to run model on ('cpu' or 'cuda')
        """
        super().__init__(**kwargs)

        self.device = torch.device(device)
        self.charge = charge
        self.mult = mult

        model_path = Path(model_path)

        if model_path.suffix == '.jpt':
            self.model = torch.jit.load(str(model_path), map_location=self.device)
            self.model.eval()

            if hasattr(self.model, 'cutoff'):
                self.cutoff = self.model.cutoff
            else:
                self.cutoff = 5.0

            # Load metadata from sidecar YAML if available
            metadata_path = model_path.with_name(model_path.stem + '_meta.yaml')
            self.metadata = {}
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.metadata = yaml.safe_load(f)

        elif model_path.suffix == '.pt':
            self.model, self.metadata = _load_pt_model(str(model_path), self.device)
            self.model.eval()
            self.cutoff = self.metadata.get('cutoff', 5.0)

        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}. Use .jpt or .pt files.")

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: List[str] = ['energy'],
        system_changes: List[str] = all_changes
    ):
        """Perform calculation."""
        super().calculate(atoms, properties, system_changes)

        positions = torch.tensor(
            self.atoms.positions, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        numbers = torch.tensor(
            self.atoms.numbers, dtype=torch.int64, device=self.device
        ).unsqueeze(0)

        charge = torch.tensor([self.charge], dtype=torch.float32, device=self.device)
        mult = torch.tensor([self.mult], dtype=torch.float32, device=self.device)

        data = {
            'coord': positions,
            'numbers': numbers,
            'charge': charge,
            'mult': mult
        }

        if 'forces' in properties:
            positions.requires_grad_(True)
            data['coord'] = positions
            output = self.model(data)
            e = output['energy'].sum()
            forces = -torch.autograd.grad(e, positions)[0]
            self.results['forces'] = forces.squeeze(0).detach().cpu().numpy()
            self.results['energy'] = e.detach().item()
        else:
            with torch.no_grad():
                output = self.model(data)
            self.results['energy'] = output['energy'].item()


class MultiFidelityCalculator:
    """Manager for multiple fidelity-specific calculators."""

    def __init__(self, model_dir: str, device: str = 'cpu'):
        self.model_dir = Path(model_dir)
        self.device = device
        self.available_fidelities = []

        # Scan for both .jpt and .pt files
        for model_file in sorted(self.model_dir.glob('*_fid*.*')):
            if model_file.suffix not in ('.jpt', '.pt'):
                continue
            fid_str = model_file.stem.split('_fid')[-1]
            try:
                fid = int(fid_str)
                if fid not in self.available_fidelities:
                    self.available_fidelities.append(fid)
            except ValueError:
                continue

        self.available_fidelities.sort()

    def get_calculator(self, fidelity: int, charge: float = 0.0, mult: float = 1.0) -> AIMNet2Calculator:
        if fidelity not in self.available_fidelities:
            raise ValueError(f"Fidelity {fidelity} not available. Available: {self.available_fidelities}")

        # Prefer .pt over .jpt
        model_files = list(self.model_dir.glob(f'*_fid{fidelity}.pt'))
        if not model_files:
            model_files = list(self.model_dir.glob(f'*_fid{fidelity}.jpt'))
        if not model_files:
            raise FileNotFoundError(f"No model found for fidelity {fidelity}")

        return AIMNet2Calculator(str(model_files[0]), charge=charge, mult=mult, device=self.device)

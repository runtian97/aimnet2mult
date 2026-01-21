"""
ASE calculators for AIMNet2 JIT models.

The JIT models have dispersion embedded, so the calculator is simple.
"""

import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class AIMNet2Calculator(Calculator):
    """
    ASE calculator for AIMNet2 JIT models.

    The JIT model includes NN + SAE + optional D3BJ dispersion.
    Simply load and use.

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
            model_path: Path to JIT model (.jpt)
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
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}. Use .jpt files.")

        self.model.eval()

        if hasattr(self.model, 'cutoff'):
            self.cutoff = self.model.cutoff
        else:
            self.cutoff = 5.0

        # Load metadata if available
        metadata_path = model_path.with_name(model_path.stem + '_meta.yaml')
        self.metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = yaml.safe_load(f)

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

        for jpt_file in self.model_dir.glob('*_fid*.jpt'):
            fid_str = jpt_file.stem.split('_fid')[-1]
            try:
                self.available_fidelities.append(int(fid_str))
            except ValueError:
                continue

        self.available_fidelities.sort()

    def get_calculator(self, fidelity: int, charge: float = 0.0, mult: float = 1.0) -> AIMNet2Calculator:
        if fidelity not in self.available_fidelities:
            raise ValueError(f"Fidelity {fidelity} not available. Available: {self.available_fidelities}")

        model_files = list(self.model_dir.glob(f'*_fid{fidelity}.jpt'))
        if not model_files:
            raise FileNotFoundError(f"No model found for fidelity {fidelity}")

        return AIMNet2Calculator(str(model_files[0]), charge=charge, mult=mult, device=self.device)

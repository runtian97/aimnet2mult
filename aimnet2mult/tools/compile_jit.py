"""
Compile trained fidelity-specific model to TorchScript JIT with SAE and Dispersion.

This script:
1. Loads the trained model checkpoint
2. Wraps it with SAE addition layer
3. Optionally adds dispersion correction (D3BJ or D4) embedded in the model
4. Compiles to TorchScript JIT format for deployment

The compiled model can be used with:
- torch.jit.load() for direct PyTorch usage
- ASE calculator for molecular dynamics
"""

import os
import yaml
import torch
import torch.nn as nn
from typing import Dict, Optional, Any


# D3BJ parameters for common functionals
# Reference: https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/dft-d3/functionalsbj
D3BJ_PARAMS = {
    'b3lyp': {'s8': 1.9889, 'a1': 0.3981, 'a2': 4.4211, 's6': 1.0},
    'pbe': {'s8': 0.7875, 'a1': 0.4289, 'a2': 4.4407, 's6': 1.0},
    'pbe0': {'s8': 1.2177, 'a1': 0.4145, 'a2': 4.8593, 's6': 1.0},
    'wb97x': {'s8': 0.0000, 'a1': 0.0000, 'a2': 5.4959, 's6': 1.0},
    'wb97m': {'s8': 0.3908, 'a1': 0.5660, 'a2': 3.1280, 's6': 1.0},
    'tpss': {'s8': 1.9435, 'a1': 0.4535, 'a2': 4.4752, 's6': 1.0},
    'bp86': {'s8': 3.2822, 'a1': 0.3946, 'a2': 4.8516, 's6': 1.0},
    'm062x': {'s8': 0.0000, 'a1': 0.0000, 'a2': 5.0580, 's6': 1.0},
    'b973c': {'s8': 1.5000, 'a1': 0.3700, 'a2': 4.1000, 's6': 1.0},
}


def get_d3bj_params(functional: str) -> Dict[str, float]:
    """Get D3BJ parameters for a functional."""
    func_lower = functional.lower().replace('-', '').replace('_', '')
    if func_lower in D3BJ_PARAMS:
        return D3BJ_PARAMS[func_lower].copy()
    for key in D3BJ_PARAMS:
        if func_lower in key or key in func_lower:
            return D3BJ_PARAMS[key].copy()
    available = ', '.join(D3BJ_PARAMS.keys())
    raise ValueError(f"Unknown functional '{functional}'. Available: {available}")


class FidelityModelWithSAE(nn.Module):
    """TorchScript-compatible wrapper with SAE only (no dispersion)."""

    __constants__ = ['fidelity_level', 'fidelity_offset']

    def __init__(
        self,
        model: nn.Module,
        sae: Dict[int, float],
        fidelity_level: int,
        fidelity_offset: int = 200,
        cutoff: float = 5.0,
        cutoff_lr: float = float('inf')
    ):
        super().__init__()
        self.model = model
        self.fidelity_level = fidelity_level
        self.fidelity_offset = fidelity_offset

        if not sae:
            raise ValueError(f"SAE dictionary is empty for fidelity {fidelity_level}")

        max_z = max(sae.keys()) if sae else 118
        tensor_size = max(max_z + 1, 119, (fidelity_level + 1) * fidelity_offset + 118)
        sae_tensor = torch.zeros(tensor_size, dtype=torch.float32)
        for z, energy in sae.items():
            shifted_z = int(z) + fidelity_level * fidelity_offset
            sae_tensor[shifted_z] = energy
        self.register_buffer('sae_tensor', sae_tensor)
        self.register_buffer('_cutoff', torch.tensor(cutoff, dtype=torch.float32))
        self.register_buffer('_cutoff_lr', torch.tensor(cutoff_lr, dtype=torch.float32))
        self.cutoff: float = cutoff
        self.cutoff_lr: float = cutoff_lr

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if '_fidelities' not in data:
            charge = data['charge']
            batch_size = 1 if charge.dim() == 0 else charge.size(0)
            fidelities = torch.full((batch_size,), self.fidelity_level, dtype=torch.long, device=charge.device)
            data = dict(data)
            data['_fidelities'] = fidelities

        output = self.model(data)

        numbers = data['numbers']
        shifted_numbers = numbers + self.fidelity_level * self.fidelity_offset
        sae_per_atom = self.sae_tensor[shifted_numbers]

        if 'mol_idx' in data:
            mol_idx = data['mol_idx']
            batch_size = int(mol_idx.max().item()) + 1
            sae_energy = torch.zeros(batch_size, dtype=sae_per_atom.dtype, device=sae_per_atom.device)
            for i in range(batch_size):
                mask = (mol_idx == i)
                sae_energy[i] = sae_per_atom[mask].sum()
        else:
            if sae_per_atom.dim() == 1:
                sae_energy = sae_per_atom.sum().unsqueeze(0)
            else:
                sae_energy = sae_per_atom.sum(dim=-1)

        output = dict(output)
        output['energy'] = output['energy'] + sae_energy
        return output


class FidelityModelWithSAEAndDispersion(nn.Module):
    """
    TorchScript-compatible wrapper with SAE and D3BJ dispersion embedded.

    The dispersion is computed using the internal DFTD3 module which is
    TorchScript compatible. The model can be loaded with torch.jit.load()
    and used directly.
    """

    __constants__ = ['fidelity_level', 'fidelity_offset', 'has_dispersion']

    def __init__(
        self,
        model: nn.Module,
        sae: Dict[int, float],
        fidelity_level: int,
        fidelity_offset: int = 200,
        cutoff: float = 5.0,
        cutoff_lr: float = float('inf'),
        dispersion_module: Optional[nn.Module] = None
    ):
        super().__init__()
        self.model = model
        self.fidelity_level = fidelity_level
        self.fidelity_offset = fidelity_offset

        if not sae:
            raise ValueError(f"SAE dictionary is empty for fidelity {fidelity_level}")

        max_z = max(sae.keys()) if sae else 118
        tensor_size = max(max_z + 1, 119, (fidelity_level + 1) * fidelity_offset + 118)
        sae_tensor = torch.zeros(tensor_size, dtype=torch.float32)
        for z, energy in sae.items():
            shifted_z = int(z) + fidelity_level * fidelity_offset
            sae_tensor[shifted_z] = energy
        self.register_buffer('sae_tensor', sae_tensor)
        self.register_buffer('_cutoff', torch.tensor(cutoff, dtype=torch.float32))
        self.register_buffer('_cutoff_lr', torch.tensor(cutoff_lr, dtype=torch.float32))
        self.cutoff: float = cutoff
        self.cutoff_lr: float = cutoff_lr

        self.dispersion = dispersion_module
        self.has_dispersion: bool = dispersion_module is not None

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if '_fidelities' not in data:
            charge = data['charge']
            batch_size = 1 if charge.dim() == 0 else charge.size(0)
            fidelities = torch.full((batch_size,), self.fidelity_level, dtype=torch.long, device=charge.device)
            data = dict(data)
            data['_fidelities'] = fidelities

        # Run model - this sets up neighbor lists needed for dispersion
        output = self.model(data)

        # Add SAE correction
        numbers = data['numbers']
        shifted_numbers = numbers + self.fidelity_level * self.fidelity_offset
        sae_per_atom = self.sae_tensor[shifted_numbers]

        if 'mol_idx' in data:
            mol_idx = data['mol_idx']
            batch_size = int(mol_idx.max().item()) + 1
            sae_energy = torch.zeros(batch_size, dtype=sae_per_atom.dtype, device=sae_per_atom.device)
            for i in range(batch_size):
                mask = (mol_idx == i)
                sae_energy[i] = sae_per_atom[mask].sum()
        else:
            if sae_per_atom.dim() == 1:
                sae_energy = sae_per_atom.sum().unsqueeze(0)
            else:
                sae_energy = sae_per_atom.sum(dim=-1)

        output = dict(output)
        output['energy'] = output['energy'] + sae_energy

        # Add dispersion if configured
        if self.has_dispersion and self.dispersion is not None:
            # Pass the data dict (with neighbor lists from model) to dispersion
            # The dispersion module adds to 'energy' key
            disp_data = dict(data)
            disp_data['energy'] = output['energy']
            disp_data = self.dispersion(disp_data)
            output['energy'] = disp_data['energy']

        return output


def compile_model(
    checkpoint_path: str,
    output_prefix: str,
    model_config: Dict,
    fidelity_level: int,
    fidelity_offset: int,
    num_fidelities: int,
    use_fidelity_readouts: bool,
    sae: Dict[int, float],
    dispersion_type: str = 'none',
    dispersion_functional: Optional[str] = None,
    dispersion_params: Optional[Dict[str, float]] = None
):
    """
    Compile a single fidelity model to TorchScript JIT with dispersion embedded.

    Args:
        checkpoint_path: Path to trained model checkpoint
        output_prefix: Output path prefix (will add _fid{N}.jpt)
        model_config: Model architecture configuration dict
        fidelity_level: Fidelity level to compile
        fidelity_offset: Atomic number offset per fidelity
        num_fidelities: Total number of fidelities
        use_fidelity_readouts: Whether model uses fidelity-specific readouts
        sae: SAE dictionary {atomic_number: energy}
        dispersion_type: 'd3bj', 'd4', or 'none'
        dispersion_functional: Functional name for predefined params (e.g., 'pbe', 'wb97m')
        dispersion_params: Custom dispersion parameters (overrides functional)
    """
    from ..config import build_module
    from ..modules import DFTD3

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint.state_dict()

    base_model = build_module(model_config)
    base_num_elements = base_model.afv.weight.shape[0]
    z_offset = fidelity_level * fidelity_offset

    # Extract weights for this fidelity
    base_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('shared_readout.') or key.startswith('fidelity_readouts.'):
            continue
        elif key == 'afv.weight':
            if value.shape[0] > z_offset + base_num_elements:
                base_state_dict[key] = value[z_offset:z_offset+base_num_elements, :].clone()
            else:
                base_state_dict[key] = value[:base_num_elements, :].clone()
        elif 'atomic_shift.shifts.weight' in key:
            target_size = base_model.outputs.atomic_shift.shifts.weight.shape[0]
            if value.shape[0] > z_offset + target_size:
                base_state_dict[key] = value[z_offset:z_offset+target_size, :].clone()
            else:
                base_state_dict[key] = value[:target_size, :].clone()
        else:
            base_state_dict[key] = value

    # Load fidelity-specific readouts
    atomic_shift_target_size = base_model.outputs.atomic_shift.shifts.weight.shape[0]
    if use_fidelity_readouts and f'fidelity_readouts.{fidelity_level}.energy_mlp.mlp.0.weight' in state_dict:
        for key in list(state_dict.keys()):
            if key.startswith(f'fidelity_readouts.{fidelity_level}.'):
                new_key = key.replace(f'fidelity_readouts.{fidelity_level}.', 'outputs.')
                value = state_dict[key]
                if 'atomic_shift.shifts.weight' in key:
                    if value.shape[0] > z_offset + atomic_shift_target_size:
                        base_state_dict[new_key] = value[z_offset:z_offset+atomic_shift_target_size, :].clone()
                    else:
                        base_state_dict[new_key] = value[:atomic_shift_target_size, :].clone()
                else:
                    base_state_dict[new_key] = value
    elif 'shared_readout.energy_mlp.mlp.0.weight' in state_dict:
        for key in list(state_dict.keys()):
            if key.startswith('shared_readout.'):
                new_key = key.replace('shared_readout.', 'outputs.')
                value = state_dict[key]
                if 'atomic_shift.shifts.weight' in key:
                    if value.shape[0] > z_offset + atomic_shift_target_size:
                        base_state_dict[new_key] = value[z_offset:z_offset+atomic_shift_target_size, :].clone()
                    else:
                        base_state_dict[new_key] = value[:atomic_shift_target_size, :].clone()
                else:
                    base_state_dict[new_key] = value

    try:
        base_model.load_state_dict(base_state_dict, strict=True)
    except RuntimeError:
        base_model.load_state_dict(base_state_dict, strict=False)

    base_model.eval()

    if isinstance(model_config, dict):
        cutoff = model_config.get('kwargs', {}).get('aev', {}).get('rc_s', 5.0)
    else:
        cutoff = 5.0

    # Build dispersion module
    dispersion_module = None
    if dispersion_type == 'd3bj':
        if dispersion_params:
            params = dispersion_params
        elif dispersion_functional:
            params = get_d3bj_params(dispersion_functional)
        else:
            raise ValueError("D3BJ requires --dispersion-functional or explicit params")

        dispersion_module = DFTD3(
            s8=params['s8'],
            a1=params['a1'],
            a2=params['a2'],
            s6=params.get('s6', 1.0),
            key_out='energy'
        )
    elif dispersion_type == 'd4':
        raise NotImplementedError(
            "D4 dispersion is not yet implemented as an embedded module. "
            "Use d3bj or handle D4 at inference time with dftd4 package."
        )

    # Create wrapped model
    if dispersion_module is not None:
        model_with_sae = FidelityModelWithSAEAndDispersion(
            base_model, sae, fidelity_level, fidelity_offset,
            cutoff=cutoff, cutoff_lr=float('inf'),
            dispersion_module=dispersion_module
        ).eval()
    else:
        model_with_sae = FidelityModelWithSAE(
            base_model, sae, fidelity_level, fidelity_offset,
            cutoff=cutoff, cutoff_lr=float('inf')
        ).eval()

    # Save paths
    if output_prefix.endswith('.pt') or output_prefix.endswith('.jpt'):
        output_prefix = output_prefix.rsplit('.', 1)[0]
    output_path = f"{output_prefix}_fid{fidelity_level}.jpt"

    try:
        scripted = torch.jit.script(model_with_sae)
        scripted.save(output_path)

        # Save metadata
        metadata = {
            'fidelity_level': fidelity_level,
            'fidelity_offset': fidelity_offset,
            'num_fidelities': num_fidelities,
            'sae': sae,
            'cutoff': cutoff,
            'dispersion': {
                'type': dispersion_type,
                'functional': dispersion_functional,
                'params': dispersion_params if dispersion_params else (
                    get_d3bj_params(dispersion_functional) if dispersion_type == 'd3bj' and dispersion_functional else None
                ),
                'embedded': dispersion_type != 'none'
            }
        }
        metadata_path = f"{output_prefix}_fid{fidelity_level}_meta.yaml"
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)

    except Exception as e:
        fallback_path = f"{output_prefix}_fid{fidelity_level}.pt"
        torch.save({
            'model_state_dict': model_with_sae.state_dict(),
            'sae': sae,
            'fidelity_level': fidelity_level,
            'fidelity_offset': fidelity_offset,
            'cutoff': cutoff,
            'dispersion_type': dispersion_type,
            'dispersion_functional': dispersion_functional,
            'eager_model': True,
            'error': str(e)
        }, fallback_path)
        raise


if __name__ == '__main__':
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description='Compile fidelity model to JIT with SAE and Dispersion')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--model', type=str, required=True, help='Path to model config')
    parser.add_argument('--output', type=str, required=True, help='Output path prefix')
    parser.add_argument('--fidelity-level', type=int, required=True, help='Fidelity level')
    parser.add_argument('--fidelity-offset', type=int, default=200, help='Fidelity offset')
    parser.add_argument('--num-fidelities', type=int, required=True, help='Number of fidelities')
    parser.add_argument('--use-fidelity-readouts', type=str, default='True', help='Use fidelity readouts')
    parser.add_argument('--sae', type=str, required=True, help='Path to SAE file')

    # Dispersion arguments
    parser.add_argument('--dispersion', type=str, default='none', choices=['none', 'd3bj', 'd4'],
                       help='Dispersion type to embed in model')
    parser.add_argument('--dispersion-functional', type=str, default=None,
                       help='Functional name for D3BJ params (e.g., pbe, b3lyp, wb97m)')
    parser.add_argument('--dispersion-s8', type=float, default=None, help='D3BJ s8 parameter')
    parser.add_argument('--dispersion-a1', type=float, default=None, help='D3BJ a1 parameter')
    parser.add_argument('--dispersion-a2', type=float, default=None, help='D3BJ a2 parameter')
    parser.add_argument('--dispersion-s6', type=float, default=1.0, help='D3BJ s6 parameter')

    args = parser.parse_args()

    model_cfg = OmegaConf.load(args.model)
    model_cfg = OmegaConf.to_container(model_cfg)

    with open(args.sae, 'r') as f:
        sae = yaml.safe_load(f)

    use_fidelity_readouts = args.use_fidelity_readouts.lower() in ('true', '1', 'yes')

    # Build custom params if provided
    dispersion_params = None
    if args.dispersion == 'd3bj' and args.dispersion_s8 is not None:
        dispersion_params = {
            's8': args.dispersion_s8,
            'a1': args.dispersion_a1,
            'a2': args.dispersion_a2,
            's6': args.dispersion_s6
        }

    compile_model(
        args.weights, args.output, model_cfg, args.fidelity_level,
        args.fidelity_offset, args.num_fidelities, use_fidelity_readouts,
        sae, args.dispersion, args.dispersion_functional, dispersion_params
    )

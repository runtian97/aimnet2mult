"""
Compile trained fidelity-specific model to TorchScript JIT with SAE.

This script:
1. Loads the trained model checkpoint
2. Wraps it with SAE addition layer
3. Compiles to TorchScript JIT format for deployment
"""

import yaml
import torch
import torch.nn as nn
import warnings
from functools import partial
from typing import Dict, Optional

from aimnet2mult.models.mixed_fidelity_aimnet2 import MixedFidelityAIMNet2
from ..utils.units import (
    HARTREE_TO_EV,
    FORCE_HARTREE_BOHR_TO_EV_ANGSTROM,
)

from torch.jit._trace import TracerWarning


class FidelityModelWithSAE(nn.Module):
    """
    Scriptable wrapper that adds SAE to model predictions.

    This wrapper preserves the model's ability to handle both:
    - 3D batched input: [batch, atoms, 3] (for direct use)
    - 2D flattened input: [total_atoms, 3] (for AIMNet2Calculator)

    The base model's prepare_input handles the format conversion.
    """

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

        # Convert SAE dict to tensor
        max_z = max(sae.keys()) if sae else 118
        tensor_size = max(max_z + 1, 119, (fidelity_level + 1) * fidelity_offset + 118)
        sae_tensor = torch.zeros(tensor_size, dtype=torch.float32)
        for z, energy in sae.items():
            shifted_z = int(z) + fidelity_level * fidelity_offset
            sae_tensor[shifted_z] = energy
        self.register_buffer('sae_tensor', sae_tensor)

        # Store cutoff values as buffers for TorchScript
        self.register_buffer('_cutoff', torch.tensor(cutoff, dtype=torch.float32))
        self.register_buffer('_cutoff_lr', torch.tensor(cutoff_lr, dtype=torch.float32))

        # Store as Python float attributes (aimnet2calc expects floats, not tensors)
        # TorchScript will preserve these as constants
        self.cutoff: float = cutoff
        self.cutoff_lr: float = cutoff_lr

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Add fidelity labels if not present
        if '_fidelities' not in data:
            # Determine batch size from charge
            charge = data['charge']
            if charge.dim() == 0:
                batch_size = 1
            else:
                batch_size = charge.size(0)

            # Create fidelity tensor
            fidelities = torch.full(
                (batch_size,),
                self.fidelity_level,
                dtype=torch.long,
                device=charge.device
            )
            # Add to data dict
            data = dict(data)  # Make a copy to avoid mutating input
            data['_fidelities'] = fidelities

        # Pass data to model (it handles prepare_input for format conversion)
        output = self.model(data)

        # Add SAE correction to energy
        # Note: Input atomic numbers are NORMAL (1-118), but SAE tensor uses SHIFTED indices
        numbers = data['numbers']
        shifted_numbers = numbers + self.fidelity_level * self.fidelity_offset
        sae_per_atom = self.sae_tensor[shifted_numbers]

        # Aggregate SAE by molecule
        if 'mol_idx' in data:
            # Flattened format from AIMNet2Calculator
            mol_idx = data['mol_idx']
            batch_size = int(mol_idx.max().item()) + 1
            sae_energy = torch.zeros(batch_size, dtype=sae_per_atom.dtype, device=sae_per_atom.device)
            for i in range(batch_size):
                mask = (mol_idx == i)
                sae_energy[i] = sae_per_atom[mask].sum()
        else:
            # Direct 3D format
            if sae_per_atom.dim() == 1:
                # Single flattened molecule
                sae_energy = sae_per_atom.sum().unsqueeze(0)
            else:
                # Batched: [batch, atoms]
                sae_energy = sae_per_atom.sum(dim=-1)

        # Add to output energy
        output = dict(output)  # Make a copy
        output['energy'] = output['energy'] + sae_energy

        return output


def compile_model_with_sae(checkpoint_path: str, output_prefix: str, model_config: Dict, fidelity_level: Optional[int] = None, train_cfg: Optional[Dict] = None):
    """
    Compile model with SAE to TorchScript.

    Args:
        checkpoint_path: Path to saved checkpoint (.pt file)
        output_prefix: Prefix for output files (will append _fid{N}.pt)
        model_config: Model architecture configuration
        fidelity_level: If specified, only compile for this fidelity. Otherwise compile all.
        train_cfg: The training configuration.
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(checkpoint.keys())

    # Extract components
    state_dict = checkpoint['model']
    num_fidelities = len(train_cfg.data.fidelity_datasets)
    fidelity_offset = train_cfg.get('fidelity_offset', 200)
    use_fidelity_readouts = train_cfg.get('use_fidelity_readouts', True)

    # Load SAE from file
    with open(args.sae, 'r') as f:
        sae_per_fidelity = {fidelity_level: yaml.safe_load(f)}

    print(f"Model configuration:")
    print(f"  Fidelities: {num_fidelities}")
    print(f"  Fidelity offset: {fidelity_offset}")
    print(f"  Fidelity readouts enabled: {use_fidelity_readouts}")
    for fid, sae_map in sae_per_fidelity.items():
        print(f"  Fidelity {fid} SAE entries: {len(sae_map)}")

    # Determine which fidelities to compile
    if fidelity_level is not None:
        fidelities_to_compile = [fidelity_level]
    else:
        fidelities_to_compile = list(range(num_fidelities))

    compiled_models = {}

    for fid in fidelities_to_compile:
        print(f"\nCompiling model for fidelity {fid}...")
        if fid not in sae_per_fidelity:
            raise ValueError(f"No SAE data available for fidelity {fid}")

        # Build a SINGLE-FIDELITY base model (no fidelity readouts) for JIT compatibility
        print(f"  Building single-fidelity base model (JIT-compatible)...")
        from ..config import build_module
        base_model = build_module(model_config)

        # Load weights from the mixed-fidelity model
        # Need to map fidelity-offset atomic numbers back to standard numbers
        # The mixed-fidelity model uses offset atomic numbers: fidelity N uses Z + N*offset
        # We need to extract rows from the offset range and pack them into standard Z range

        # Get the expected size from the base model
        base_num_elements = base_model.afv.weight.shape[0]  # Usually 64 or 119
        z_offset = fid * fidelity_offset

        print(f"  Base model expects {base_num_elements} elements, extracting from offset {z_offset}")

        base_state_dict = {}
        for key, value in state_dict.items():
            # Skip fidelity-specific readout layers
            if key.startswith('shared_readout.') or key.startswith('fidelity_readouts.'):
                continue
            # Handle embedding weights (afv.weight) - extract fidelity-specific slice
            elif key == 'afv.weight':
                # Extract rows for this fidelity's atomic numbers (z_offset to z_offset+base_num_elements)
                base_state_dict[key] = value[z_offset:z_offset+base_num_elements, :].clone()
            # Copy all other weights directly (outputs will be handled separately)
            else:
                base_state_dict[key] = value

        # For outputs, use the fidelity-specific readout if available
        if f'fidelity_readouts.{fid}.energy_mlp.mlp.0.weight' in state_dict:
            print(f"  Using fidelity-specific readout for fidelity {fid}")
            for key in list(state_dict.keys()):
                if key.startswith(f'fidelity_readouts.{fid}.'):
                    new_key = key.replace(f'fidelity_readouts.{fid}.', 'outputs.')
                    value = state_dict[key]
                    # Extract fidelity-specific slice for atomic shifts
                    if 'atomic_shift.shifts.weight' in key:
                        base_state_dict[new_key] = value[z_offset:z_offset+base_num_elements, :].clone()
                    else:
                        base_state_dict[new_key] = value
        elif 'shared_readout.energy_mlp.mlp.0.weight' in state_dict:
            print(f"  Using shared readout")
            for key in list(state_dict.keys()):
                if key.startswith('shared_readout.'):
                    new_key = key.replace('shared_readout.', 'outputs.')
                    value = state_dict[key]
                    # Extract fidelity-specific slice for atomic shifts
                    if 'atomic_shift.shifts.weight' in key:
                        base_state_dict[new_key] = value[z_offset:z_offset+base_num_elements, :].clone()
                    else:
                        base_state_dict[new_key] = value

        base_model.load_state_dict(base_state_dict)
        base_model.eval()

        # Extract cutoff from model config
        if isinstance(model_config, dict):
            if 'kwargs' in model_config and 'aev' in model_config['kwargs']:
                cutoff = model_config['kwargs']['aev'].get('rc_s', 5.0)
            elif 'aev' in model_config:
                cutoff = model_config['aev'].get('rc_s', 5.0)
            else:
                cutoff = 5.0  # Default
        else:
            cutoff = 5.0  # Default

        # Wrap with SAE
        print(f"  Wrapping model with SAE (cutoff={cutoff})...")
        model_with_sae = FidelityModelWithSAE(
            base_model,  # Pass base model, not mixed-fidelity model
            sae_per_fidelity[fid],
            fid,
            fidelity_offset,
            cutoff=cutoff,
            cutoff_lr=float('inf')
        ).eval()

        # Save as TorchScript JIT model
        if output_prefix.endswith('.pt'):
            output_path = output_prefix[:-3] + f'_fid{fid}.jpt'
        elif output_prefix.endswith('.jpt'):
            output_path = output_prefix[:-4] + f'_fid{fid}.jpt'
        else:
            output_path = output_prefix + f'_fid{fid}.jpt'

        print(f"  Saving TorchScript model to {output_path}")
        try:
            # Script the model directly - no eager format needed
            scripted = torch.jit.script(model_with_sae)
            scripted.save(output_path)
            print(f"  ✓ Saved TorchScript model (loadable with torch.jit.load)")
        except Exception as e:
            print(f"  ERROR: Could not create TorchScript version: {e}")
            print(f"  Falling back to eager model format...")
            # Fallback: save eager model with metadata
            save_dict = {
                'model_state_dict': model_with_sae.state_dict(),
                'sae': sae_per_fidelity[fid],
                'fidelity_level': fid,
                'fidelity_offset': fidelity_offset,
                'num_fidelities': num_fidelities,
                'use_fidelity_readouts': False,  # Base model doesn't use readouts
                'model_config': model_config,
                'cutoff': cutoff,
                'cutoff_lr': float('inf'),
                'eager_model': True
            }
            torch.save(save_dict, output_path)

        compiled_models[fid] = model_with_sae

    print("\nCompilation complete!")
    print(f"Created {len(compiled_models)} JIT model(s)")
    return compiled_models


if __name__ == '__main__':
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description='Compile fidelity-specific model to JIT with SAE')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights to load.')
    parser.add_argument('--model', type=str, required=True, help='Path to model definition file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save final model weights.')
    parser.add_argument('--fidelity-level', type=int, required=True, help='Fidelity level to compile.')
    parser.add_argument('--fidelity-offset', type=int, required=True, help='Offset for fidelity levels.')
    parser.add_argument('--num-fidelities', type=int, required=True, help='Total number of fidelities.')
    parser.add_argument('--use-fidelity-readouts', type=bool, default=True, help='Whether fidelity readouts are used.')
    parser.add_argument('--sae', type=str, required=True, help='Path to SAE file.')
    parser.add_argument('--train-config', type=str, required=False, help='Path to the training config file (deprecated).')

    args = parser.parse_args()

    # Load model config
    model_cfg = OmegaConf.load(args.model)
    model_cfg = OmegaConf.to_container(model_cfg)

    # Create minimal train config from args
    train_cfg = OmegaConf.create({
        'data': {'fidelity_datasets': {i: '' for i in range(args.num_fidelities)}},
        'fidelity_offset': args.fidelity_offset,
        'use_fidelity_readouts': args.use_fidelity_readouts
    })

    # Compile
    compile_model_with_sae(args.weights, args.output, model_cfg, args.fidelity_level, train_cfg)

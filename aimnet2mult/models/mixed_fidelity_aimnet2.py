"""
Mixed-Fidelity AIMNet2 model that processes batches with molecules from different fidelities.

Key features:
- Processes batches where each molecule has its own fidelity label
- Uses shifted atomic numbers (already offset in data)
- Supports fidelity-dependent readouts
- Atomic embeddings are fidelity-specific via atomic number offsets
"""

import torch
import torch.nn as nn
from typing import Dict

from .aimnet2 import AIMNet2


class MixedFidelityAIMNet2(AIMNet2):
    """
    AIMNet2 variant that handles mixed-fidelity batches.

    The key insight: atomic numbers are already offset per fidelity in the data,
    so the model automatically uses fidelity-specific embeddings. We just need
    to handle fidelity-dependent readouts.
    """

    def __init__(
        self,
        base_model_config: Dict,
        num_fidelities: int = 3,
        fidelity_offset: int = 200,
        use_fidelity_readouts: bool = True,
        **kwargs
    ):
        """
        Args:
            base_model_config: Base AIMNet2 configuration
            num_fidelities: Number of fidelities
            fidelity_offset: Offset between fidelities for atomic numbers
            use_fidelity_readouts: If True, use separate readouts per fidelity
        """
        # Calculate max atomic number needed
        max_z_base = 128
        max_z_total = num_fidelities * fidelity_offset + max_z_base

        # Build base model using aimnet.config.build_module
        from ..config import build_module

        # Extract parameters from config
        if isinstance(base_model_config, dict) and 'kwargs' in base_model_config:
            cfg = base_model_config['kwargs']
        else:
            cfg = base_model_config

        # Build outputs separately without building full model
        outputs_cfg = cfg['outputs']
        outputs = build_module(outputs_cfg)

        # Initialize base model with config parameters, passing max_z
        super().__init__(
            aev=cfg['aev'],
            nfeature=cfg['nfeature'],
            d2features=cfg['d2features'],
            ncomb_v=cfg['ncomb_v'],
            hidden=cfg['hidden'],
            aim_size=cfg['aim_size'],
            outputs=outputs,
            num_charge_channels=cfg.get('num_charge_channels', 1),
            max_z=max_z_total + 1
        )

        self.num_fidelities = num_fidelities
        self.fidelity_offset = fidelity_offset
        self.use_fidelity_readouts = use_fidelity_readouts

        # Create fidelity-specific readouts if requested
        if use_fidelity_readouts:
            self._create_fidelity_readouts(base_model_config)

    def _create_fidelity_readouts(self, config):
        """Create separate readout layers for each fidelity."""
        from ..config import build_module
        from ..modules import AtomicShift

        # Store original readout
        self.shared_readout = self.outputs

        # Create fidelity-specific readouts
        self.fidelity_readouts = nn.ModuleDict()

        # Calculate max atomic number for all fidelities
        max_z_base = 128
        max_z_total = self.num_fidelities * self.fidelity_offset + max_z_base

        for fid in range(self.num_fidelities):
            # Build outputs directly without building full model
            if isinstance(config, dict) and 'kwargs' in config:
                outputs_cfg = config['kwargs']['outputs']
            else:
                outputs_cfg = config['outputs']
            fid_outputs = build_module(outputs_cfg)

            # Expand atomic shift modules to handle shifted atomic numbers
            for key, module in fid_outputs.items():
                if isinstance(module, AtomicShift):
                    # Expand the shifts embedding to accommodate all fidelities
                    old_shifts = module.shifts
                    old_size = old_shifts.weight.shape[0]
                    new_size = max_z_total + 1

                    # Create new embedding with expanded size
                    new_shifts = nn.Embedding(new_size, old_shifts.weight.shape[1], padding_idx=0)

                    # Copy existing weights and initialize new ones
                    with torch.no_grad():
                        nn.init.zeros_(new_shifts.weight)
                        new_shifts.weight[:old_size] = old_shifts.weight

                    module.shifts = new_shifts

            # Convert dict to ModuleDict before assigning
            self.fidelity_readouts[str(fid)] = nn.ModuleDict(fid_outputs)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        fidelities: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for mixed-fidelity batch.

        Args:
            data: Input dict with 'coord', 'numbers', 'charge', 'mult'
                  Note: 'numbers' should already have fidelity offsets applied
                  Can also contain '_fidelities' key with fidelity labels
            fidelities: Tensor of shape [batch_size] with fidelity index for each molecule
                        (can be None if provided in data dict)

        Returns:
            Output dict with predictions
        """
        # Extract fidelities from data dict if present
        if fidelities is None and '_fidelities' in data:
            fidelities = data.pop('_fidelities')

        # If still no fidelities provided, assume all are fidelity 0
        if fidelities is None:
            # Determine batch size based on input format
            if 'mol_idx' in data:
                # Flattened format: batch size is max(mol_idx) + 1
                batch_size = int(data['mol_idx'].max().item()) + 1
            else:
                # 3D batched format: batch size is first dimension of coord
                batch_size = data['coord'].shape[0]
            fidelities = torch.zeros(batch_size, dtype=torch.long, device=data['coord'].device)

        # Call parent forward (this handles all the message passing)
        # Parent will call self.outputs which we can override
        if self.use_fidelity_readouts:
            # Temporarily replace outputs with a wrapper
            orig_outputs = self.outputs
            self.outputs = nn.ModuleDict()  # Empty - we'll handle readouts manually

            # Run parent forward (gets us to the AIM representation)
            data = super().forward(data)

            # Restore outputs and apply fidelity-specific readouts
            self.outputs = orig_outputs
            data = self._apply_fidelity_readouts(data, fidelities)
        else:
            # Use parent's forward as-is (will use self.outputs)
            data = super().forward(data)

        return data

    def _apply_fidelity_readouts(self, data: Dict[str, torch.Tensor], fidelities: torch.Tensor):
        """
        Apply fidelity-specific readouts to molecules.

        Optimized implementation: only compute readouts for molecules belonging to each
        fidelity, avoiding redundant computation. Each molecule is processed by exactly
        one readout head (its own fidelity's), not all readout heads.

        This produces identical gradients to the naive approach because gradients only
        flow through the readout that produces the output used in the loss.
        """
        batch_size = fidelities.size(0)
        device = fidelities.device

        # Initialize result with input data
        result = dict(data)

        # Output keys that readouts produce
        output_keys = {'energy', 'charges', 'spin_charges'}

        # Pre-allocate output tensors (will be filled per-fidelity)
        # We need to determine shapes from a sample readout pass
        output_tensors = {}

        # Get unique fidelities present in this batch (avoid computing unused fidelities)
        unique_fidelities = torch.unique(fidelities)

        # Process each fidelity separately - only compute for molecules of that fidelity
        for fid in unique_fidelities:
            fid_int = fid.item()

            # Get indices of molecules belonging to this fidelity
            mask = (fidelities == fid)
            indices = torch.where(mask)[0]

            if len(indices) == 0:
                continue

            # Extract subset of data for this fidelity's molecules
            fid_data = {}
            for key, value in data.items():
                if isinstance(value, torch.Tensor) and value.dim() > 0 and value.size(0) == batch_size:
                    # Batch-dimension tensor - extract subset
                    fid_data[key] = value[indices]
                else:
                    # Non-batch tensor (e.g., scalar) - keep as is
                    fid_data[key] = value

            # Apply this fidelity's readout chain to only its molecules
            readouts = self.fidelity_readouts[str(fid_int)]
            for key, module in readouts.items():
                fid_data = module(fid_data)

            # Place results back into the correct positions in output tensors
            for key in output_keys:
                if key not in fid_data:
                    continue

                fid_output = fid_data[key]

                # Handle scalar outputs
                if fid_output.dim() == 0:
                    result[key] = fid_output
                    continue

                # Initialize output tensor on first encounter
                if key not in output_tensors:
                    # Determine full output shape
                    output_shape = (batch_size,) + fid_output.shape[1:]
                    output_tensors[key] = torch.zeros(output_shape, dtype=fid_output.dtype, device=device)

                # Place this fidelity's outputs at the correct indices
                output_tensors[key][indices] = fid_output

        # Add output tensors to result
        for key, tensor in output_tensors.items():
            result[key] = tensor

        return result

    def _extract_subset(self, data: Dict[str, torch.Tensor], indices: torch.Tensor):
        """Extract subset of batch for specific indices."""
        subset = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor) and len(value.shape) > 0:
                # Check if this is a batch-sized tensor
                if value.shape[0] == indices.shape[0]:
                    # This is likely a per-molecule tensor (already correct size)
                    subset[key] = value
                elif value.shape[0] > indices.shape[0]:
                    # This is a larger batch tensor - extract subset
                    subset[key] = value[indices]
                else:
                    # Some other sized tensor - keep as is
                    subset[key] = value
            else:
                # Scalar tensor or other type  - keep as is
                subset[key] = value

        return subset

    def _combine_outputs(self, original_data, outputs_by_fidelity, batch_size, device):
        """Combine outputs from different fidelities back into batch order."""
        # Initialize output tensors
        combined = dict(original_data)

        # Find all output keys
        output_keys = set()
        for fid, (indices, fid_data) in outputs_by_fidelity.items():
            output_keys.update(fid_data.keys())

        # Combine each output
        for key in output_keys:
            if key.startswith('_'):
                # Skip internal keys
                continue

            # Determine output shape
            first_fid = list(outputs_by_fidelity.keys())[0]
            first_indices, first_data = outputs_by_fidelity[first_fid]

            if key not in first_data or not isinstance(first_data[key], torch.Tensor):
                continue

            first_tensor = first_data[key]

            # Check if tensor is scalar (0-dimensional)
            if len(first_tensor.shape) == 0:
                # Scalar tensor - just use the first one (they should all be the same)
                combined[key] = first_tensor
                continue

            # Collect all tensors in the correct batch order
            tensors_by_index = {}
            for fid, (indices, fid_data) in outputs_by_fidelity.items():
                if key in fid_data:
                    for i, idx in enumerate(indices):
                        tensors_by_index[idx.item()] = fid_data[key][i:i+1]

            # Concatenate in correct order to preserve gradient flow
            if tensors_by_index:
                ordered_tensors = [tensors_by_index[i] for i in sorted(tensors_by_index.keys())]
                combined[key] = torch.cat(ordered_tensors, dim=0)

        return combined

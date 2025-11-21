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

        # Build a temporary model to get the architecture
        temp_model = build_module(base_model_config)

        # Extract parameters from config
        if isinstance(base_model_config, dict) and 'kwargs' in base_model_config:
            cfg = base_model_config['kwargs']
        else:
            cfg = base_model_config

        # Initialize base model with config parameters
        super().__init__(
            aev=cfg['aev'],
            nfeature=cfg['nfeature'],
            d2features=cfg['d2features'],
            ncomb_v=cfg['ncomb_v'],
            hidden=cfg['hidden'],
            aim_size=cfg['aim_size'],
            outputs=dict(temp_model.outputs),  # Convert ModuleDict to dict
            num_charge_channels=cfg.get('num_charge_channels', 1)
        )

        # Replace AFV embedding with larger table to accommodate all fidelities
        old_afv_shape = self.afv.weight.shape
        new_afv_size = max_z_total + 1
        new_afv = nn.Embedding(new_afv_size, old_afv_shape[1], padding_idx=0)

        # Copy over existing weights for base atomic numbers
        with torch.no_grad():
            # Initialize with small random values
            nn.init.normal_(new_afv.weight, mean=0, std=0.01)
            new_afv.weight[:old_afv_shape[0]] = self.afv.weight

        self.afv = new_afv
        self.max_z = max_z_total

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
            # Build a new model to get fresh output layers for this fidelity
            temp_model = build_module(config)
            fid_outputs = temp_model.outputs

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

            self.fidelity_readouts[str(fid)] = fid_outputs

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

        To preserve gradient flow for forces computation, we apply ALL fidelity readouts
        to ALL molecules, then select the appropriate output based on fidelity labels.
        This ensures that gradients flow through all coordinates.
        """
        batch_size = fidelities.size(0)
        device = fidelities.device

        # Apply each fidelity's readouts to the entire batch
        all_outputs = {}
        for fid in range(self.num_fidelities):
            fid_data = dict(data)  # Copy the input data

            # Apply this fidelity's readout chain
            readouts = self.fidelity_readouts[str(fid)]
            for key, module in readouts.items():
                fid_data = module(fid_data)

            all_outputs[fid] = fid_data

        # Select the appropriate output for each molecule based on its fidelity
        # This preserves gradients for all molecules
        result = dict(data)

        # Find all output keys that were produced by readouts
        output_keys = set()
        for fid_data in all_outputs.values():
            output_keys.update(fid_data.keys())

        # For each output key, select the right value for each molecule
        # Only process known final outputs
        allowed_keys = {'energy'}  # Only process total energy
        for key in output_keys:
            if key not in allowed_keys:
                # Skip everything except allowed final outputs
                continue

            # Check if this key exists in any fidelity output
            has_output = any(key in all_outputs[fid] for fid in range(self.num_fidelities))
            if not has_output:
                continue

            # Get a sample output to determine shape
            sample_output = None
            for fid in range(self.num_fidelities):
                if key in all_outputs[fid]:
                    sample_output = all_outputs[fid][key]
                    break

            if sample_output is None:
                continue

            # Handle scalar outputs (like total energy)
            if sample_output.dim() == 0:
                # For scalar outputs, just use the output from the first fidelity
                # (they should be the same for single-molecule batches)
                result[key] = sample_output
                continue

            # Align tensors so the leading dimension matches the batch size.
            def _align_to_batch(tensor: torch.Tensor) -> torch.Tensor:
                try:
                    return tensor.expand(batch_size, *tensor.shape[1:])
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Unexpected leading dimension for key '{key}': got {tensor.size(0)}, "
                        f"expected 1 or {batch_size}"
                    ) from exc

            aligned_sample = _align_to_batch(sample_output)

            # For per-molecule or per-atom outputs, select based on fidelity
            # Stack outputs from all fidelities
            stacked_outputs = []
            for fid in range(self.num_fidelities):
                if key in all_outputs[fid]:
                    stacked_outputs.append(_align_to_batch(all_outputs[fid][key]))
                else:
                    # If this fidelity doesn't have this output, create zeros
                    stacked_outputs.append(torch.zeros_like(aligned_sample))

            # Stack along a new dimension: [num_fidelities, batch_size, ...]
            stacked = torch.stack(stacked_outputs, dim=0)

            # Select the appropriate fidelity for each molecule
            # fidelities shape: [batch_size]
            # We need to select stacked[fidelity[i], i, ...] for each i

            # Create index tensors
            batch_indices = torch.arange(batch_size, device=device)

            # Use advanced indexing to select the right fidelity for each molecule
            try:
                if len(stacked.shape) == 2:
                    # Shape: [num_fidelities, batch_size] -> [batch_size]
                    result[key] = stacked[fidelities, batch_indices]
                elif len(stacked.shape) == 3:
                    # Shape: [num_fidelities, batch_size, feature_dim] -> [batch_size, feature_dim]
                    result[key] = stacked[fidelities, batch_indices, :]
                else:
                    # General case: [num_fidelities, batch_size, ...] -> [batch_size, ...]
                    # This uses gather, which preserves gradients
                    result[key] = torch.index_select(
                        stacked.view(self.num_fidelities * batch_size, *stacked.shape[2:]),
                        dim=0,
                        index=fidelities * batch_size + batch_indices
                    ).view(batch_size, *stacked.shape[2:])
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Failed to select fidelity-specific output for key '{key}' with "
                    f"stacked shape {tuple(stacked.shape)}, sample shape {tuple(sample_output.shape)}, "
                    f"batch_size={batch_size}, fidelities shape={tuple(fidelities.shape)}"
                ) from exc

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

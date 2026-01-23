"""Metrics for mixed-fidelity training."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict

import ignite.distributed as idist
import torch
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class RegMultiMetric(Metric):
    """
    Multi-property regression metric with configurable unit conversions.

    Supports the same configuration format as base aimnet2, with additional
    support for masked properties (for mixed-fidelity training where some
    molecules may not have all labels).
    """

    def __init__(self, cfg: Dict, loss_fn=None):
        """
        Initialize the metric.

        Args:
            cfg: Configuration dictionary mapping property names to config dicts.
                 Each property config can have:
                 - abbr: Abbreviation for logging (e.g., 'E' for energy)
                 - scale: Unit conversion factor (default: 1.0)
                 - peratom: Whether to normalize by number of atoms (default: False)
                 - mult: Dimensionality multiplier (default: 1.0)
            loss_fn: Optional loss function to track validation loss
        """
        super().__init__()
        self.cfg = cfg
        self.loss_fn = loss_fn

    def attach_loss(self, loss_fn):
        """Attach a loss function for validation loss tracking."""
        self.loss_fn = loss_fn

    @reinit__is_reduced
    def reset(self):
        """Reset metric state."""
        super().reset()
        # Use None initially - will be initialized on first update with correct device
        self._device = None
        self.data = defaultdict(lambda: defaultdict(lambda: None))
        self.atoms = 0.0
        self.samples = 0.0
        self.loss = defaultdict(float)

    def _update_one(self, key: str, pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor = None) -> None:
        """
        Update metric for a single property.

        Accumulates on GPU to avoid CPU-GPU sync. Transfer to CPU only in compute().

        Args:
            key: Property name
            pred: Predicted values
            true: True values
            mask: Optional mask for valid values (for mixed-fidelity training)
        """
        # Track device for later use
        if self._device is None:
            self._device = pred.device

        e = true - pred

        # Apply mask if provided
        if mask is not None:
            # Ensure mask has same shape as error - use expand instead of loop
            if mask.dim() < e.dim():
                mask = mask.view(mask.shape + (1,) * (e.dim() - mask.dim()))
            e = e * mask
            n_valid = mask.sum()  # Keep as tensor on GPU
        else:
            n_valid = torch.tensor(e.numel(), device=pred.device, dtype=torch.float32)

        # Flatten error for accumulation
        if pred.ndim > true.ndim:
            e = e.view(pred.shape[0], -1)
        else:
            e = e.view(-1)

        # Accumulate on GPU (use float32 for MPS compatibility)
        d = self.data[key]
        sum_abs = e.abs().sum(-1).to(dtype=torch.float32)
        sum_sq = e.pow(2).sum(-1).to(dtype=torch.float32)
        sum_true = true.sum().to(dtype=torch.float32)
        sum_sq_true = true.pow(2).sum().to(dtype=torch.float32)

        # Initialize or accumulate (keep on GPU)
        if d['sum_abs_err'] is None:
            d['sum_abs_err'] = sum_abs
            d['sum_sq_err'] = sum_sq
            d['sum_true'] = sum_true
            d['sum_sq_true'] = sum_sq_true
            d['n_valid'] = n_valid.float()
        else:
            d['sum_abs_err'] = d['sum_abs_err'] + sum_abs
            d['sum_sq_err'] = d['sum_sq_err'] + sum_sq
            d['sum_true'] = d['sum_true'] + sum_true
            d['sum_sq_true'] = d['sum_sq_true'] + sum_sq_true
            d['n_valid'] = d['n_valid'] + n_valid.float()

    @reinit__is_reduced
    def update(self, output) -> None:
        """Update metric with a batch of predictions and targets."""
        y_pred, y_true = output
        if y_pred is None:
            return

        # Update each property
        for k in y_pred:
            if k.startswith('_') or k not in y_true:
                continue

            # Check for mask (for mixed-fidelity training with missing labels)
            mask_key = f"{k}_mask"
            mask = y_true.get(mask_key, None)

            with torch.no_grad():
                self._update_one(k, y_pred[k].detach(), y_true[k].detach(), mask)

            b = y_true[k].shape[0]

        self.samples += b

        # Track number of atoms
        if '_natom' in y_pred:
            _n = y_pred['_natom']
            if _n.numel() > 1:
                self.atoms += _n.sum().item()
            else:
                self.atoms += y_pred['numbers'].shape[0] * y_pred['numbers'].shape[1]
        elif 'numbers' in y_pred:
            self.atoms += y_pred['numbers'].shape[0] * y_pred['numbers'].shape[1]

        # Update loss if loss_fn is provided
        if self.loss_fn is not None:
            with torch.no_grad():
                loss_d = self.loss_fn(y_pred, y_true)
                for k, loss in loss_d.items():
                    if isinstance(loss, torch.Tensor):
                        if loss.numel() > 1:
                            loss = loss.mean()
                        loss = loss.item()
                    self.loss[k] += loss * b

    def compute(self):
        """Compute final metrics across all accumulated batches.

        Transfers accumulated GPU tensors to CPU only here, avoiding
        GPU-CPU sync during training loop.
        """
        if self.samples == 0:
            return {}

        # Transfer GPU tensors to CPU for final computation
        # This is the ONLY place we do GPU->CPU transfer
        for k1, v1 in self.data.items():
            for k2, v2 in v1.items():
                if isinstance(v2, torch.Tensor):
                    self.data[k1][k2] = v2.to(dtype=torch.float64, device='cpu')

        # Synchronize across distributed processes
        if idist.get_world_size() > 1:
            self.atoms = idist.all_reduce(self.atoms)
            self.samples = idist.all_reduce(self.samples)
            for k, loss in self.loss.items():
                self.loss[k] = idist.all_reduce(loss)
            for k1, v1 in self.data.items():
                for k2, v2 in v1.items():
                    self.data[k1][k2] = idist.all_reduce(v2)

        self._is_reduced = True

        # Compute metrics
        ret = dict()
        for k in self.data:
            if k not in self.cfg:
                continue

            cfg = self.cfg[k]

            # Determine normalization (per-atom or per-molecule)
            _n = self.atoms if cfg.get('peratom', False) else self.samples
            _n *= cfg.get('mult', 1.0)

            name = k
            abbr = cfg['abbr']
            v = self.data[name]

            # Use n_valid if available (for masked properties), otherwise use _n
            n_samples = v.get('n_valid', _n)
            if isinstance(n_samples, torch.Tensor):
                n_samples = n_samples.item()
            if n_samples == 0:
                continue

            m = dict()
            m['mae'] = v['sum_abs_err'] / n_samples
            m['rmse'] = (v['sum_sq_err'] / n_samples).sqrt()
            m['r2'] = 1.0 - v['sum_sq_err'] / (v['sum_sq_true'] - (v['sum_true'].pow(2)) / n_samples)

            # Apply scale factor for unit conversion
            for metric_name, metric_val in m.items():
                if metric_name in ('mae', 'rmse'):
                    metric_val *= cfg.get('scale', 1.0)
                metric_val = metric_val.tolist()
                if isinstance(metric_val, list):
                    for ii, vv in enumerate(metric_val):
                        ret[f'{abbr}_{metric_name}_{ii}'] = vv
                else:
                    ret[f'{abbr}_{metric_name}'] = metric_val

        # Add loss metrics
        if len(self.loss):
            for k, loss in self.loss.items():
                if not k.endswith('loss'):
                    k = k + '_loss'
                ret[k] = loss / self.samples

        logging.info(str(ret))

        return ret


# Legacy functions for backward compatibility
class PropertyMAE(Metric):
    """Mean Absolute Error for a specific property."""

    def __init__(self, property_key: str, output_transform=lambda x: x, device=None):
        self.property_key = property_key
        self._sum_abs_error = None
        self._num_examples = None
        super(PropertyMAE, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_abs_error = 0.0
        self._num_examples = 0
        super(PropertyMAE, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y_true = output

        if self.property_key not in y_true:
            return

        if self.property_key not in y_pred:
            return

        pred = y_pred[self.property_key]
        true = y_true[self.property_key]

        # Check for mask
        mask_key = f"{self.property_key}_mask"
        if mask_key in y_true:
            mask = y_true[mask_key].squeeze(-1)  # Remove last dim if present
            # Expand mask to match prediction shape if needed
            while mask.dim() < pred.dim():
                mask = mask.unsqueeze(-1)
            # Only compute error where mask is 1
            abs_error = torch.abs(pred - true) * mask
            num_samples = mask.sum().item()
            if num_samples > 0:
                self._sum_abs_error += abs_error.sum().item()
                self._num_examples += num_samples
        else:
            abs_error = torch.abs(pred - true)
            self._sum_abs_error += abs_error.sum().item()
            self._num_examples += abs_error.numel()

    @sync_all_reduce("_sum_abs_error", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            return 0.0
        return self._sum_abs_error / self._num_examples


class PropertyRMSE(Metric):
    """Root Mean Squared Error for a specific property."""

    def __init__(self, property_key: str, output_transform=lambda x: x, device=None):
        self.property_key = property_key
        self._sum_squared_error = None
        self._num_examples = None
        super(PropertyRMSE, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_squared_error = 0.0
        self._num_examples = 0
        super(PropertyRMSE, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y_true = output

        if self.property_key not in y_true:
            return

        if self.property_key not in y_pred:
            return

        pred = y_pred[self.property_key]
        true = y_true[self.property_key]

        # Check for mask
        mask_key = f"{self.property_key}_mask"
        if mask_key in y_true:
            mask = y_true[mask_key].squeeze(-1)  # Remove last dim if present
            # Expand mask to match prediction shape if needed
            while mask.dim() < pred.dim():
                mask = mask.unsqueeze(-1)
            squared_error = ((pred - true) ** 2) * mask
            num_samples = mask.sum().item()
            if num_samples > 0:
                self._sum_squared_error += squared_error.sum().item()
                self._num_examples += num_samples
        else:
            squared_error = (pred - true) ** 2
            self._sum_squared_error += squared_error.sum().item()
            self._num_examples += squared_error.numel()

    @sync_all_reduce("_sum_squared_error", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            return 0.0
        return (self._sum_squared_error / self._num_examples) ** 0.5


class PerMoleculeLoss(Metric):
    """Average loss per molecule."""

    def __init__(self, loss_fn, output_transform=lambda x: x, device=None):
        self.loss_fn = loss_fn
        self._sum_loss = None
        self._num_molecules = None
        super(PerMoleculeLoss, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_loss = 0.0
        self._num_molecules = 0
        super(PerMoleculeLoss, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y_true = output
        loss_out = self.loss_fn(y_pred, y_true)
        loss = loss_out.get("loss") if isinstance(loss_out, dict) else loss_out

        batch_size = next(iter(y_true.values())).shape[0]
        self._sum_loss += loss.item() * batch_size
        self._num_molecules += batch_size

    @sync_all_reduce("_sum_loss", "_num_molecules")
    def compute(self):
        if self._num_molecules == 0:
            return 0.0
        return self._sum_loss / self._num_molecules


def create_metrics(loss_fn, properties: list[str], device=None):
    """
    Create a dictionary of metrics for validation (legacy interface).

    Args:
        loss_fn: Loss function to use for computing validation loss
        properties: List of properties to track (e.g., ['energy', 'forces', 'charges'])
        device: Device to put metrics on

    Returns:
        Dictionary of metric name -> Metric object
    """
    metrics = {}

    # Always track validation loss
    metrics["loss"] = PerMoleculeLoss(loss_fn, device=device)

    # Track MAE and RMSE for each property
    for prop in properties:
        metrics[f"{prop}_mae"] = PropertyMAE(prop, device=device)
        metrics[f"{prop}_rmse"] = PropertyRMSE(prop, device=device)

    return metrics


def format_metrics(metrics_dict: dict, unit_conversions: dict = None, unit_labels: dict = None) -> str:
    """
    Format metrics for logging (legacy interface).

    Args:
        metrics_dict: Dictionary of metric name -> value
        unit_conversions: Optional dictionary of property -> conversion factor to apply
        unit_labels: Optional dictionary of property -> unit label string

    Returns:
        Formatted string for logging
    """
    if unit_conversions is None:
        # Default unit conversions: eV to kcal/mol (matching aimnet2)
        # 1 eV = 23.06 kcal/mol
        ev_to_kcal = 23.06054783061903
        unit_conversions = {
            "energy": ev_to_kcal,
            "forces": ev_to_kcal,  # eV/Å to kcal/mol/Å uses same factor
            "charges": 1.0,  # Already in elementary charge units
            "spin_charges": 1.0,
        }

    if unit_labels is None:
        unit_labels = {
            "energy": "kcal/mol",
            "forces": "kcal/mol/Å",
            "charges": "e",
            "spin_charges": "e",
        }

    lines = []

    # First show loss
    if "loss" in metrics_dict:
        lines.append(f"  Loss: {metrics_dict['loss']:.6f}")

    # Then show each property
    for prop in ["energy", "forces", "charges", "spin_charges"]:
        mae_key = f"{prop}_mae"
        rmse_key = f"{prop}_rmse"

        if mae_key in metrics_dict:
            conv = unit_conversions.get(prop, 1.0)
            mae_val = metrics_dict[mae_key] * conv
            rmse_val = metrics_dict[rmse_key] * conv

            # Get unit label
            unit = unit_labels.get(prop, "")

            lines.append(f"  {prop.capitalize():12s} MAE: {mae_val:8.4f} {unit:12s} RMSE: {rmse_val:8.4f} {unit}")

    return "\n".join(lines)

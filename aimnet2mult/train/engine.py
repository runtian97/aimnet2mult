"""Ignite engine builders for mixed-fidelity training."""

from __future__ import annotations

from typing import Dict, Tuple

from ignite.engine import Engine
from torch import nn
import torch


from .utils import prepare_batch  # noqa: E402

__all__ = ["create_trainer", "create_evaluator", "compute_batch_rmse"]


def create_trainer(model: nn.Module, optimizer, loss_fn, device, use_base_model: bool = False):
    """Create the Ignite training engine with fidelity-aware batches.

    Args:
        model: The model to train
        optimizer: The optimizer
        loss_fn: The loss function
        device: The device to use
        use_base_model: If True, don't pass fidelities to the model (base AIMNet2 mode)
    """

    def _update(engine, batch: Tuple[Dict, Dict, "torch.Tensor"]):
        model.train()
        optimizer.zero_grad()

        x, y, fidelities = batch
        x = prepare_batch(x, device=device)
        y = prepare_batch(y, device=device)

        # Only pass fidelities if using mixed-fidelity model
        if not use_base_model:
            fidelities = fidelities.to(device)
            x["_fidelities"] = fidelities

        pred = model(x)
        loss_out = loss_fn(pred, y)
        total_loss = loss_out.get("loss") if isinstance(loss_out, dict) else loss_out
        if total_loss is None:
            raise ValueError("Loss function returned a dict without 'loss' key.")

        total_loss.backward()
        # Gradient clipping for stability (same as aimnet2)
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.4)
        optimizer.step()

        # Store pred and y for metrics computation
        engine.state.last_pred = pred
        engine.state.last_y = y

        # Return loss value for WandB logging (matches aimnet2 behavior)
        return total_loss.item()

    return Engine(_update)


def compute_batch_rmse(pred: Dict, y_true: Dict) -> Dict:
    """Compute RMSE for each property on a single batch.

    Called on-demand by logging handlers, not every iteration.
    """
    rmse = {}
    with torch.no_grad():
        for key in ['energy', 'forces', 'charges', 'spin_charges']:
            if key not in pred or key not in y_true:
                continue

            # Check for mask (mixed-fidelity training)
            mask_key = f"{key}_mask"
            mask = y_true.get(mask_key, None)

            error = (pred[key] - y_true[key]).pow(2)

            if mask is not None:
                # Apply mask - use view instead of while loop
                if mask.dim() < error.dim():
                    mask = mask.view(mask.shape + (1,) * (error.dim() - mask.dim()))
                error = error * mask
                n_valid = mask.sum().item()
                if n_valid == 0:
                    continue
                mse = error.sum().item() / n_valid
            else:
                mse = error.mean().item()

            rmse[key] = mse ** 0.5

    return rmse


def create_evaluator(model: nn.Module, device, use_base_model: bool = False):
    """Create the Ignite evaluation engine.

    Args:
        model: The model to evaluate
        device: The device to use
        use_base_model: If True, don't pass fidelities to the model (base AIMNet2 mode)
    """

    def _inference(engine, batch: Tuple[Dict, Dict, "torch.Tensor"]):
        model.eval()
        with torch.no_grad():
            x, y, fidelities = batch
            x = prepare_batch(x, device=device)
            y = prepare_batch(y, device=device)

            # Only pass fidelities if using mixed-fidelity model
            if not use_base_model:
                fidelities = fidelities.to(device)
                x["_fidelities"] = fidelities

            pred = model(x)
        return pred, y

    return Engine(_inference)

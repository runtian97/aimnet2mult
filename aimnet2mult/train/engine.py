"""Ignite engine builders for mixed-fidelity training."""

from __future__ import annotations

from typing import Dict, Tuple

from ignite.engine import Engine
from torch import nn
import torch


from .utils import prepare_batch  # noqa: E402

__all__ = ["create_trainer", "create_evaluator"]


def create_trainer(model: nn.Module, optimizer, loss_fn, device):
    """Create the Ignite training engine with fidelity-aware batches."""

    def _update(engine, batch: Tuple[Dict, Dict, "torch.Tensor"]):
        model.train()
        optimizer.zero_grad()

        x, y, fidelities = batch
        x = prepare_batch(x, device=device)
        y = prepare_batch(y, device=device)
        fidelities = fidelities.to(device)
        x["_fidelities"] = fidelities

        pred = model(x)
        loss_out = loss_fn(pred, y)
        total_loss = loss_out.get("loss") if isinstance(loss_out, dict) else loss_out
        if total_loss is None:
            raise ValueError("Loss function returned a dict without 'loss' key.")

        total_loss.backward()
        optimizer.step()

        # Return pred and y for metrics computation
        return pred, y

    return Engine(_update)


def create_evaluator(model: nn.Module, device):

    def _inference(engine, batch: Tuple[Dict, Dict, "torch.Tensor"]):
        model.eval()
        with torch.no_grad():
            x, y, fidelities = batch
            x = prepare_batch(x, device=device)
            y = prepare_batch(y, device=device)
            fidelities = fidelities.to(device)
            x["_fidelities"] = fidelities
            pred = model(x)
        return pred, y

    return Engine(_inference)

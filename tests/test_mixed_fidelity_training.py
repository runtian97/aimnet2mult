"""Integration test for mixed-fidelity training with missing labels.

Verifies that training with two fidelity datasets — where fidelity 1 is
missing ``charges`` and ``spin_charges`` — produces finite (non-NaN/inf)
loss and metric values throughout training and validation.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from aimnet2mult.train.configuration import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TRAIN_CONFIG_PATH,
    _convert_fidelity_keys_to_int,
)
from aimnet2mult.train.runner import run_training

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Small dataset: 4 light elements
ELEMENTS_SMALL = [1, 6, 7, 8]  # H, C, N, O
SAE_SMALL = {1: -0.5, 6: -37.0, 7: -54.0, 8: -75.0}

# Larger dataset: 10 elements spanning rows 1-4 of the periodic table
ELEMENTS_LARGE = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17]  # H, B, C, N, O, F, Si, P, S, Cl
SAE_LARGE = {
    1: -0.5,
    5: -24.5,
    6: -37.8,
    7: -54.5,
    8: -75.0,
    9: -99.7,
    14: -289.0,
    15: -340.7,
    16: -397.5,
    17: -460.1,
}


def _random_numbers(
    natoms: int, n_mol: int, elements: List[int], rng: np.random.Generator
) -> np.ndarray:
    """Return (n_mol, natoms) int32 array of atomic numbers drawn from *elements*."""
    return rng.choice(elements, size=(n_mol, natoms)).astype(np.int32)


def _write_h5(
    path: str,
    sizes: List[int],
    n_mol: int,
    include_charges: bool,
    elements: List[int],
    sae_values: Dict[int, float],
    rng: np.random.Generator,
) -> None:
    """Create a fake HDF5 dataset in the expected size-grouped format."""
    sorted_elements = sorted(elements)
    sae_arr = np.array([sae_values[z] for z in sorted_elements])

    with h5py.File(path, "w") as f:
        for natoms in sizes:
            grp = f.create_group(f"{natoms:03d}")
            numbers = _random_numbers(natoms, n_mol, sorted_elements, rng)
            coords = rng.standard_normal((n_mol, natoms, 3)).astype(np.float32)

            # Energy = sum-of-SAE + small noise
            elem_idx = np.searchsorted(sorted_elements, numbers)
            energy = sae_arr[elem_idx].sum(axis=-1) + rng.standard_normal(n_mol) * 0.01
            energy = energy.astype(np.float64)

            forces = rng.standard_normal((n_mol, natoms, 3)).astype(np.float32) * 0.1
            charge = np.zeros(n_mol, dtype=np.float32)
            mult = np.ones(n_mol, dtype=np.int32)

            grp.create_dataset("coord", data=coords)
            grp.create_dataset("numbers", data=numbers)
            grp.create_dataset("energy", data=energy)
            grp.create_dataset("forces", data=forces)
            grp.create_dataset("charge", data=charge)
            grp.create_dataset("mult", data=mult)

            if include_charges:
                charges = rng.standard_normal((n_mol, natoms)).astype(np.float32) * 0.1
                spin_charges = rng.standard_normal((n_mol, natoms)).astype(np.float32) * 0.01
                grp.create_dataset("charges", data=charges)
                grp.create_dataset("spin_charges", data=spin_charges)


def _write_sae(path: str, sae_values: Dict[int, float]) -> None:
    with open(path, "w") as f:
        yaml.dump({int(k): float(v) for k, v in sae_values.items()}, f)


def _make_datasets(
    tmp_path: Path,
    elements: List[int],
    sae_values: Dict[int, float],
    sizes: List[int],
    n_mol: int,
    seed: int = 42,
) -> dict:
    """Create two fake HDF5 datasets and matching SAE files."""
    rng = np.random.default_rng(seed)

    fid0_path = str(tmp_path / "fid0.h5")
    fid1_path = str(tmp_path / "fid1.h5")
    sae0_path = str(tmp_path / "sae_fid0.yaml")
    sae1_path = str(tmp_path / "sae_fid1.yaml")

    _write_h5(fid0_path, sizes, n_mol, include_charges=True,
              elements=elements, sae_values=sae_values, rng=rng)
    _write_h5(fid1_path, sizes, n_mol, include_charges=False,
              elements=elements, sae_values=sae_values, rng=rng)
    _write_sae(sae0_path, sae_values)
    _write_sae(sae1_path, sae_values)

    return {
        "fid0": fid0_path,
        "fid1": fid1_path,
        "sae0": sae0_path,
        "sae1": sae1_path,
        "tmp_path": tmp_path,
    }


def _build_train_cfg(ds: dict, *, epochs: int = 2, batches_per_epoch: int = 5):
    """Build a merged train config for testing."""
    model_cfg = OmegaConf.load(str(DEFAULT_MODEL_PATH))
    train_cfg = OmegaConf.load(str(DEFAULT_TRAIN_CONFIG_PATH))

    checkpoint_dir = str(ds["tmp_path"] / "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    overrides = OmegaConf.create(
        {
            "run_name": "test_run",
            "project_name": "test",
            "data": {
                "fidelity_datasets": {0: ds["fid0"], 1: ds["fid1"]},
                "fidelity_weights": {0: 1.0, 1: 1.0},
                "sae": {"energy": {"files": {0: ds["sae0"], 1: ds["sae1"]}, "mode": "linreg"}},
                "val_fraction": 0.2,
                "samplers": {
                    "train": {"kwargs": {"batch_size": 16, "batch_mode": "molecules", "shuffle": True, "batches_per_epoch": batches_per_epoch, "sampling_strategy": "weighted"}},
                    "val": {"kwargs": {"batch_size": 16, "batch_mode": "molecules", "shuffle": False, "batches_per_epoch": batches_per_epoch, "sampling_strategy": "uniform"}},
                },
            },
            "trainer": {"epochs": epochs},
            "wandb": None,
            "checkpoint": {
                "dirname": checkpoint_dir,
                "filename_prefix": "test",
                "kwargs": {"n_saved": 1, "require_empty": False},
            },
            "scheduler": None,
        }
    )

    train_cfg = OmegaConf.merge(train_cfg, overrides)
    _convert_fidelity_keys_to_int(train_cfg)
    return model_cfg, train_cfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_datasets_small(tmp_path: Path):
    """Small datasets: 4 elements (H,C,N,O), 2 size groups, 50 molecules each."""
    return _make_datasets(
        tmp_path, ELEMENTS_SMALL, SAE_SMALL,
        sizes=[3, 5], n_mol=50,
    )


@pytest.fixture()
def tmp_datasets_large(tmp_path: Path):
    """Larger datasets: 10 elements, 4 size groups, 200 molecules each."""
    return _make_datasets(
        tmp_path, ELEMENTS_LARGE, SAE_LARGE,
        sizes=[3, 5, 8, 12], n_mol=200,
    )


# ---------------------------------------------------------------------------
# Tests — small datasets
# ---------------------------------------------------------------------------


def test_training_with_missing_labels(tmp_datasets_small):
    """Two-epoch training run must complete with all-finite metrics."""
    ds = tmp_datasets_small
    save_path = str(ds["tmp_path"] / "model.pt")

    model_cfg, train_cfg = _build_train_cfg(ds, epochs=2)

    run_training(
        OmegaConf.to_container(model_cfg, resolve=True),
        OmegaConf.to_container(train_cfg, resolve=True),
        load_path=None,
        save_path=save_path,
    )

    # Model file saved
    assert os.path.isfile(save_path), "Saved model file not found"

    # Checkpoint is loadable and well-formed
    ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
    assert "model" in ckpt, "Checkpoint missing 'model' key"
    state_dict = ckpt["model"]
    assert len(state_dict) > 0, "State dict is empty"

    # All saved parameters are finite
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor) and param.is_floating_point():
            assert torch.isfinite(param).all(), (
                f"Non-finite values in saved parameter '{name}'"
            )


def test_metrics_finite_with_missing_labels(tmp_datasets_small):
    """Directly build model + loaders and verify train/val metrics are finite."""
    _run_metrics_check(tmp_datasets_small)


# ---------------------------------------------------------------------------
# Tests — larger datasets (more elements, more size groups, more molecules)
# ---------------------------------------------------------------------------


def test_training_large_dataset_with_missing_labels(tmp_datasets_large):
    """End-to-end training with 10 elements, 4 size groups, 200 mol/group."""
    ds = tmp_datasets_large
    save_path = str(ds["tmp_path"] / "model.pt")

    model_cfg, train_cfg = _build_train_cfg(ds, epochs=2, batches_per_epoch=10)

    run_training(
        OmegaConf.to_container(model_cfg, resolve=True),
        OmegaConf.to_container(train_cfg, resolve=True),
        load_path=None,
        save_path=save_path,
    )

    assert os.path.isfile(save_path), "Saved model file not found"

    ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
    assert "model" in ckpt
    for name, param in ckpt["model"].items():
        if isinstance(param, torch.Tensor) and param.is_floating_point():
            assert torch.isfinite(param).all(), (
                f"Non-finite values in saved parameter '{name}'"
            )


def test_metrics_finite_large_dataset(tmp_datasets_large):
    """Metrics check with 10 elements, 4 size groups, 200 mol/group."""
    _run_metrics_check(tmp_datasets_large, batches_per_epoch=10)


# ---------------------------------------------------------------------------
# Shared metric-check helper
# ---------------------------------------------------------------------------


def _run_metrics_check(ds_fixture: dict, batches_per_epoch: int = 5):
    """Build model + loaders, run 1 epoch, and assert all metrics are finite."""
    from ignite.engine import Events

    from aimnet2mult.config import build_module
    from aimnet2mult.data import create_mixed_fidelity_loaders
    from aimnet2mult.models.mixed_fidelity_aimnet2 import MixedFidelityAIMNet2
    from aimnet2mult.modules import Forces
    from aimnet2mult.train.engine import create_evaluator, create_trainer
    from aimnet2mult.train.metrics import RegMultiMetric

    model_cfg, train_cfg = _build_train_cfg(
        ds_fixture, epochs=1, batches_per_epoch=batches_per_epoch,
    )

    model = MixedFidelityAIMNet2(
        base_model_config=OmegaConf.to_container(model_cfg, resolve=True),
        num_fidelities=2,
        fidelity_offset=train_cfg.get("fidelity_offset", 200),
        use_fidelity_readouts=train_cfg.get("use_fidelity_readouts", True),
    )
    model = Forces(model)
    device = torch.device("cpu")

    train_loader, val_loader = create_mixed_fidelity_loaders(train_cfg)

    loss_fn = build_module(OmegaConf.to_container(train_cfg.loss, resolve=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = create_trainer(model, optimizer, loss_fn, device, use_base_model=False)
    evaluator = create_evaluator(model, device, use_base_model=False)

    # Attach RegMultiMetric to the evaluator
    metrics_kwargs = OmegaConf.to_container(train_cfg.metrics.get("kwargs", {}))
    val_metric = RegMultiMetric(**metrics_kwargs)
    val_metric.attach_loss(loss_fn)
    val_metric.attach(evaluator, "multi")

    # Collect train losses per iteration
    train_losses: list[float] = []

    @trainer.on(Events.ITERATION_COMPLETED)
    def _record_loss(engine):
        train_losses.append(engine.state.output)

    # Run one epoch of training
    trainer.run(train_loader, max_epochs=1)

    # Run validation
    evaluator.run(val_loader)

    # --- Assertions on train loss ---
    assert len(train_losses) > 0, "No training iterations ran"
    for i, loss_val in enumerate(train_losses):
        assert math.isfinite(loss_val), f"Train loss at iteration {i} is {loss_val}"

    # --- Assertions on validation metrics ---
    val_metrics = evaluator.state.metrics
    assert len(val_metrics) > 0, "No validation metrics computed"

    for metric_name, metric_val in val_metrics.items():
        if isinstance(metric_val, (int, float)):
            assert math.isfinite(metric_val), (
                f"Validation metric '{metric_name}' is {metric_val}"
            )

    # Energy and forces metrics must always be present
    metric_keys = set(val_metrics.keys())
    assert any(k.startswith("E_") for k in metric_keys), (
        f"No energy metrics found. Keys: {metric_keys}"
    )
    assert any(k.startswith("F_") for k in metric_keys), (
        f"No forces metrics found. Keys: {metric_keys}"
    )

    # Charges/spin_charges metrics should be present and finite (or zero)
    for prefix in ("q_", "s_"):
        matching = [k for k in metric_keys if k.startswith(prefix)]
        if matching:
            for k in matching:
                v = val_metrics[k]
                assert isinstance(v, (int, float)) and math.isfinite(v), (
                    f"Metric '{k}' is {v}"
                )


# ---------------------------------------------------------------------------
# Diagnostic tests — probe specific NaN triggers
# ---------------------------------------------------------------------------


def _build_model_and_loaders(ds_fixture, batches_per_epoch=5, use_production_optimizer=False):
    """Helper: build model, loaders, loss, optimizer matching production config."""
    from aimnet2mult.config import build_module
    from aimnet2mult.data import create_mixed_fidelity_loaders
    from aimnet2mult.models.mixed_fidelity_aimnet2 import MixedFidelityAIMNet2
    from aimnet2mult.modules import Forces
    from aimnet2mult.train.utils import get_optimizer

    model_cfg, train_cfg = _build_train_cfg(
        ds_fixture, epochs=1, batches_per_epoch=batches_per_epoch,
    )

    model = MixedFidelityAIMNet2(
        base_model_config=OmegaConf.to_container(model_cfg, resolve=True),
        num_fidelities=2,
        fidelity_offset=train_cfg.get("fidelity_offset", 200),
        use_fidelity_readouts=train_cfg.get("use_fidelity_readouts", True),
    )
    model = Forces(model)
    device = torch.device("cpu")

    train_loader, val_loader = create_mixed_fidelity_loaders(train_cfg)
    loss_fn = build_module(OmegaConf.to_container(train_cfg.loss, resolve=True))

    if use_production_optimizer:
        # Mirrors the real training config: RAdam with per-group LRs
        optimizer = get_optimizer(model, train_cfg.optimizer)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    return model, train_loader, val_loader, loss_fn, optimizer, device, train_cfg


def test_nan_diagnostic_forward_pass(tmp_datasets_large):
    """Check that every forward pass produces finite outputs (energy, charges, forces)."""
    model, train_loader, _, loss_fn, optimizer, device, _ = \
        _build_model_and_loaders(tmp_datasets_large, batches_per_epoch=10)

    from aimnet2mult.train.utils import prepare_batch

    model.train()
    for i, (x, y, fidelities) in enumerate(train_loader):
        x = prepare_batch(x, device=device)
        y = prepare_batch(y, device=device)
        fidelities = fidelities.to(device)
        x["_fidelities"] = fidelities

        pred = model(x)

        # Check every output tensor for NaN
        for key, val in pred.items():
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                assert torch.isfinite(val).all(), (
                    f"NaN/Inf in pred['{key}'] at batch {i}, "
                    f"fidelities={fidelities.unique().tolist()}, "
                    f"shape={val.shape}, "
                    f"min={val.min().item():.4g}, max={val.max().item():.4g}"
                )

        # Check loss components individually
        loss_out = loss_fn(pred, y)
        for key, val in loss_out.items():
            if isinstance(val, torch.Tensor):
                assert torch.isfinite(val).all(), (
                    f"NaN/Inf in loss component '{key}' at batch {i}, "
                    f"fidelities={fidelities.unique().tolist()}, "
                    f"val={val.item():.6g}"
                )

        # Check backward pass for NaN gradients
        loss_out["loss"].backward()
        for name, param in model.named_parameters():
            if param.grad is not None and param.is_floating_point():
                if not torch.isfinite(param.grad).all():
                    nan_count = (~torch.isfinite(param.grad)).sum().item()
                    pytest.fail(
                        f"NaN/Inf gradient in '{name}' at batch {i}, "
                        f"fidelities={fidelities.unique().tolist()}, "
                        f"{nan_count}/{param.grad.numel()} non-finite entries"
                    )
        optimizer.step()
        optimizer.zero_grad()


def test_nan_diagnostic_coulomb_stress(tmp_datasets_large):
    """Stress test: inject molecules with very close atoms to provoke LRCoulomb NaN."""
    from aimnet2mult.config import build_module
    from aimnet2mult.models.mixed_fidelity_aimnet2 import MixedFidelityAIMNet2
    from aimnet2mult.modules import Forces, LRCoulomb
    from aimnet2mult.train.utils import prepare_batch

    ds = tmp_datasets_large
    model_cfg = OmegaConf.load(str(DEFAULT_MODEL_PATH))

    model = MixedFidelityAIMNet2(
        base_model_config=OmegaConf.to_container(model_cfg, resolve=True),
        num_fidelities=2,
        fidelity_offset=200,
        use_fidelity_readouts=True,
    )
    model = Forces(model)
    model.eval()

    device = torch.device("cpu")

    # Create a batch with very close atoms (d_ij ~ 0.3 Å)
    batch_size = 4
    natoms = 5
    coords = torch.randn(batch_size, natoms, 3) * 0.3  # Very compact molecules
    numbers = torch.tensor([[1, 6, 7, 8, 6]] * batch_size, dtype=torch.long)
    charge = torch.zeros(batch_size)
    mult = torch.ones(batch_size, dtype=torch.long)
    fidelities = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    # Shift numbers for fidelity 1
    fid1_mask = fidelities.unsqueeze(1).expand_as(numbers) == 1
    atom_mask = numbers > 0
    numbers = numbers.clone()
    numbers[fid1_mask & atom_mask] += 200

    x = {"coord": coords, "numbers": numbers, "charge": charge, "mult": mult,
         "_fidelities": fidelities}
    x = prepare_batch(x, device=device)

    with torch.no_grad():
        pred = model(x)

    # Check outputs
    for key in ("energy", "charges", "forces"):
        if key in pred:
            val = pred[key]
            assert torch.isfinite(val).all(), (
                f"NaN/Inf in pred['{key}'] with close atoms, "
                f"min={val.min().item():.4g}, max={val.max().item():.4g}"
            )


def test_nan_diagnostic_production_optimizer(tmp_datasets_large):
    """Train with production optimizer config (RAdam + per-group LRs) for 20 iterations."""
    from ignite.engine import Events
    from aimnet2mult.train.engine import create_trainer
    from aimnet2mult.train.utils import prepare_batch

    model, train_loader, _, loss_fn, _, device, train_cfg = \
        _build_model_and_loaders(tmp_datasets_large, batches_per_epoch=20)

    # Build production optimizer: RAdam with discriminative LR
    optimizer = torch.optim.RAdam(
        [
            {"params": [], "lr": 5e-5, "weight_decay": 0.0},   # embeddings
            {"params": [], "lr": 5e-5, "weight_decay": 0.0},   # shifts
            {"params": [], "lr": 1e-4, "weight_decay": 1e-6},  # default
        ],
        lr=1e-4,
        weight_decay=1e-6,
    )
    # Assign params to groups
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "afv.weight" in name:
            optimizer.param_groups[0]["params"].append(param)
        elif "atomic_shift" in name and "shifts.weight" in name:
            optimizer.param_groups[1]["params"].append(param)
        else:
            optimizer.param_groups[2]["params"].append(param)

    # Remove empty groups
    optimizer.param_groups[:] = [g for g in optimizer.param_groups if g["params"]]

    trainer = create_trainer(model, optimizer, loss_fn, device, use_base_model=False)

    train_losses: list[float] = []

    @trainer.on(Events.ITERATION_COMPLETED)
    def _record(engine):
        train_losses.append(engine.state.output)

    trainer.run(train_loader, max_epochs=1)

    assert len(train_losses) > 0
    for i, loss_val in enumerate(train_losses):
        assert math.isfinite(loss_val), (
            f"NaN at iteration {i} with production RAdam + per-group LRs, loss={loss_val}"
        )


def test_missing_label_regularization(tmp_datasets_large):
    """Verify that missing_label_regularization produces a non-zero loss for missing labels."""
    from aimnet2mult.train.loss import peratom_loss_fn

    device = torch.device("cpu")
    y_pred = {
        "charges": torch.randn(4, 5, device=device),
        "_natom": torch.tensor(5, device=device),
    }
    # y_true has NO charges key — simulates a fidelity-1 batch
    y_true = {
        "energy": torch.randn(4, device=device),
    }

    # Without regularization: loss is 0
    loss_noreg = peratom_loss_fn(y_pred, y_true, key_pred="charges", key_true="charges",
                                  missing_label_regularization=0.0)
    assert loss_noreg.item() == 0.0

    # With regularization: loss > 0 (penalizes predictions toward zero)
    loss_reg = peratom_loss_fn(y_pred, y_true, key_pred="charges", key_true="charges",
                                missing_label_regularization=0.01)
    assert loss_reg.item() > 0.0
    assert math.isfinite(loss_reg.item())

    # Gradient flows back to predictions
    y_pred["charges"].requires_grad_(True)
    loss_reg2 = peratom_loss_fn(y_pred, y_true, key_pred="charges", key_true="charges",
                                 missing_label_regularization=0.01)
    loss_reg2.backward()
    assert y_pred["charges"].grad is not None
    assert torch.isfinite(y_pred["charges"].grad).all()


def test_nan_diagnostic_fidelity1_only_batch(tmp_datasets_large):
    """Ensure a batch containing ONLY fidelity-1 molecules (no charge labels) is finite."""
    from aimnet2mult.config import build_module
    from aimnet2mult.data import create_mixed_fidelity_loaders
    from aimnet2mult.models.mixed_fidelity_aimnet2 import MixedFidelityAIMNet2
    from aimnet2mult.modules import Forces
    from aimnet2mult.train.utils import prepare_batch

    ds = tmp_datasets_large
    model_cfg, train_cfg = _build_train_cfg(ds, epochs=1, batches_per_epoch=20)

    model = MixedFidelityAIMNet2(
        base_model_config=OmegaConf.to_container(model_cfg, resolve=True),
        num_fidelities=2,
        fidelity_offset=200,
        use_fidelity_readouts=True,
    )
    model = Forces(model)
    device = torch.device("cpu")

    train_loader, _ = create_mixed_fidelity_loaders(train_cfg)
    loss_fn = build_module(OmegaConf.to_container(train_cfg.loss, resolve=True))
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)

    model.train()
    fid1_batch_found = False

    for x, y, fidelities in train_loader:
        x = prepare_batch(x, device=device)
        y = prepare_batch(y, device=device)
        fidelities = fidelities.to(device)
        x["_fidelities"] = fidelities

        unique_fids = fidelities.unique().tolist()

        pred = model(x)
        loss_out = loss_fn(pred, y)
        total_loss = loss_out["loss"]

        assert torch.isfinite(total_loss), (
            f"Loss is {total_loss.item()} for batch with fidelities={unique_fids}, "
            f"y_keys={list(y.keys())}"
        )

        total_loss.backward()

        # Check for NaN gradients
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                pytest.fail(
                    f"NaN gradient in '{name}' for fidelities={unique_fids}"
                )

        optimizer.step()
        optimizer.zero_grad()

        if 1 in unique_fids and 0 not in unique_fids:
            fid1_batch_found = True
            # Verify charges/spin_charges NOT in y (missing labels)
            assert "charges" not in y, "Expected charges missing from fidelity-1 batch"
            assert "spin_charges" not in y, "Expected spin_charges missing from fidelity-1 batch"

    # We should have encountered at least one fidelity-1-only batch
    assert fid1_batch_found, (
        "No fidelity-1-only batch found — test inconclusive. "
        "Increase batches_per_epoch or reduce dataset size."
    )

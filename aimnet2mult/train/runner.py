"""Training orchestration for mixed-fidelity AIMNet2."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.multiprocessing as mp
from ignite import distributed as idist
from omegaconf import OmegaConf

from ..data import create_mixed_fidelity_loaders
from .engine import create_evaluator, create_trainer

from ..config import build_module  # noqa: E402
from ..modules import Forces  # noqa: E402
from .utils import get_optimizer, get_scheduler, unwrap_module, setup_wandb  # noqa: E402
from ..models.mixed_fidelity_aimnet2 import MixedFidelityAIMNet2  # noqa: E402

__all__ = ["run_training"]

try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass


def run_training(model_cfg, train_cfg, load_path: Optional[str], save_path: Optional[str]):
    """Entrypoint that spins up training on single- or multi-GPU setups."""
    num_gpus = torch.cuda.device_count()
    logging.info("Starting training using %d GPU(s)", num_gpus)
    if num_gpus == 0:
        logging.warning("No GPU available. Training will run on CPU.")

    if num_gpus > 1:
        logging.info("Using DDP training.")
        with idist.Parallel(backend="nccl", nproc_per_node=num_gpus) as parallel:
            parallel.run(_train_impl, model_cfg, train_cfg, load_path, save_path)
    else:
        _train_impl(0, model_cfg, train_cfg, load_path, save_path)


def _train_impl(local_rank, model_cfg, train_cfg, load_path, save_path):
    if local_rank == 0:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    model_cfg = OmegaConf.create(model_cfg)
    train_cfg = OmegaConf.create(train_cfg)

    _force_training = "forces" in train_cfg.data.y
    logging.info("Building mixed-fidelity AIMNet2 model...")
    model = MixedFidelityAIMNet2(
        base_model_config=OmegaConf.to_container(model_cfg, resolve=True),
        num_fidelities=len(train_cfg.data.fidelity_datasets),
        fidelity_offset=train_cfg.get("fidelity_offset", 200),
        use_fidelity_readouts=train_cfg.get("use_fidelity_readouts", True),
    )
    if _force_training:
        model = Forces(model)

    if torch.cuda.device_count() > 0:
        device = torch.device("cuda")
        model = model.to(device)
        model = idist.auto_model(model)
    else:
        device = torch.device("cpu")
        model = model.to(device)

    if load_path is not None:
        logging.info("Loading pretrained weights from %s", load_path)
        checkpoint = torch.load(load_path, map_location=device)

        # Handle both old format (bare state_dict) and new format (dict with 'model' key)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            # Old format or bare state_dict
            state_dict = checkpoint

        # For mixed-fidelity models, manually handle embedding expansion
        # Don't let pretrained embeddings overwrite the expanded ones
        unwrapped = unwrap_module(model)
        if isinstance(unwrapped, MixedFidelityAIMNet2):
            # Get current embedding sizes (already expanded)
            current_afv_size = unwrapped.afv.weight.shape[0]

            # If state_dict has smaller embeddings, expand them before loading
            if 'afv.weight' in state_dict:
                pretrained_afv = state_dict['afv.weight']
                pretrained_size = pretrained_afv.shape[0]

                if pretrained_size < current_afv_size:
                    logging.info(f"Expanding afv embedding from {pretrained_size} to {current_afv_size}")
                    # Create expanded embedding and copy pretrained weights
                    expanded_afv = torch.zeros(current_afv_size, pretrained_afv.shape[1])
                    expanded_afv[:pretrained_size] = pretrained_afv
                    state_dict['afv.weight'] = expanded_afv

            # Handle atomic shift embeddings similarly
            for key in list(state_dict.keys()):
                if 'atomic_shift.shifts.weight' in key:
                    # Extract the corresponding module from the current model
                    parts = key.split('.')
                    current_module = unwrapped
                    for part in parts[:-1]:  # Navigate to the shifts module
                        if hasattr(current_module, part):
                            current_module = getattr(current_module, part)
                        elif isinstance(current_module, nn.ModuleDict) and part in current_module:
                            current_module = current_module[part]

                    if hasattr(current_module, 'weight'):
                        current_size = current_module.weight.shape[0]
                        pretrained_shifts = state_dict[key]
                        pretrained_size = pretrained_shifts.shape[0]

                        if pretrained_size < current_size:
                            logging.info(f"Expanding {key} from {pretrained_size} to {current_size}")
                            expanded_shifts = torch.zeros(current_size, pretrained_shifts.shape[1])
                            expanded_shifts[:pretrained_size] = pretrained_shifts
                            state_dict[key] = expanded_shifts

        result = unwrapped.load_state_dict(state_dict, strict=False)
        logging.info(result)

    logging.info("Creating mixed-fidelity data loaders...")
    train_loader, val_loader = create_mixed_fidelity_loaders(train_cfg)

    logging.info("Setting up optimizer...")
    optimizer = get_optimizer(model, train_cfg.optimizer)
    if torch.cuda.device_count() > 0:
        optimizer = idist.auto_optim(optimizer)

    scheduler = None
    if train_cfg.scheduler is not None:
        scheduler = get_scheduler(optimizer, train_cfg.scheduler)

    logging.info("Creating loss function...")
    loss = build_module(OmegaConf.to_container(train_cfg.loss, resolve=True))

    logging.info("Building training engine...")
    trainer = create_trainer(model, optimizer, loss, device)
    validator = create_evaluator(model, device)

    _attach_events(trainer, validator, optimizer, scheduler, train_cfg, val_loader, model, loss)

    # Setup wandb logging if configured
    if local_rank == 0 and train_cfg.wandb is not None:
        logging.info("Setting up Weights & Biases logging...")
        setup_wandb(train_cfg, model_cfg, model, trainer, validator, optimizer)

    total_epochs = train_cfg.trainer.epochs
    logging.info("Starting training for %d epochs", total_epochs)
    logging.info("=" * 80)
    trainer.run(train_loader, max_epochs=total_epochs)
    logging.info("Training completed successfully!")
    if idist.get_local_rank() == 0:
        if save_path is not None:
            final_path = save_path
        else:
            checkpoint_cfg = getattr(train_cfg, "checkpoint", None)
            if checkpoint_cfg is None:
                raise ValueError("Checkpoint configuration missing and no --save path provided.")
            final_path = os.path.join(
                checkpoint_cfg.dirname,
                f"{checkpoint_cfg.filename_prefix}_final.pt",
            )
        logging.info("Saving final model to %s", final_path)

        # Save checkpoint with model weights only (simpler format for transfer learning)
        checkpoint_dict = {
            "model": unwrap_module(model).state_dict(),
        }

        torch.save(checkpoint_dict, final_path)


class TerminateOnLowLR:
    """Handler to terminate training when learning rate drops below a threshold."""

    def __init__(self, optimizer, low_lr=1e-5):
        self.low_lr = low_lr
        self.optimizer = optimizer

    def __call__(self, engine):
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr < self.low_lr:
            logging.info(f"Terminating training: LR ({current_lr:.2e}) < threshold ({self.low_lr:.2e})")
            engine.terminate()


def _attach_events(trainer, validator, optimizer, scheduler, train_cfg, val_loader, model, loss_fn):
    from ignite.engine import Events
    from ignite.handlers import ProgressBar, TerminateOnNan
    from .metrics import create_metrics, format_metrics

    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    def log_lr(engine):
        lr = optimizer.param_groups[0]["lr"]
        logging.info("Epoch %d - LR: %.6f", engine.state.epoch, lr)

    trainer.add_event_handler(Events.EPOCH_STARTED, log_lr)

    if idist.get_local_rank() == 0:
        pbar = ProgressBar()
        pbar.attach(trainer, event_name=Events.ITERATION_COMPLETED(every=100))

    # Attach metrics to both trainer and validator
    metrics_cfg = getattr(train_cfg, "metrics", None)
    if metrics_cfg is not None:
        # Support both new RegMultiMetric (cfg-based) and legacy interface
        if hasattr(metrics_cfg, "class"):
            # New cfg-based approach (like aimnet2)
            import importlib

            # Parse class path
            class_path = metrics_cfg["class"]
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            metric_class = getattr(module, class_name)

            # Get kwargs
            kwargs = OmegaConf.to_container(metrics_cfg.get("kwargs", {}))

            # Instantiate metrics for validator
            val_metric = metric_class(**kwargs)
            val_metric.attach_loss(loss_fn)
            val_metric.attach(validator, "metrics")

            # Instantiate separate metrics for trainer
            train_metric = metric_class(**kwargs)
            train_metric.attach_loss(loss_fn)
            train_metric.attach(trainer, "metrics")

            # Log validation metrics after validation (RegMultiMetric handles formatting)
            def log_val_metrics(engine):
                metrics_dict = engine.state.metrics.get("metrics", {})
                if metrics_dict:
                    logging.info("Validation Metrics - Epoch %d:\n%s", trainer.state.epoch, str(metrics_dict))

            validator.add_event_handler(Events.COMPLETED, log_val_metrics)

            # Log training metrics after each epoch
            def log_train_metrics(engine):
                metrics_dict = engine.state.metrics.get("metrics", {})
                if metrics_dict:
                    logging.info("Training Metrics - Epoch %d:\n%s", engine.state.epoch, str(metrics_dict))

            trainer.add_event_handler(Events.EPOCH_COMPLETED, log_train_metrics)

        elif metrics_cfg.get("enabled", True):
            # Legacy interface (backward compatibility)
            properties = train_cfg.data.y if hasattr(train_cfg.data, "y") else []

            # Create metrics for validator
            val_metrics = create_metrics(loss_fn, properties, device="cpu")
            for name, metric in val_metrics.items():
                metric.attach(validator, name)

            # Create separate metrics for trainer
            train_metrics = create_metrics(loss_fn, properties, device="cpu")
            for name, metric in train_metrics.items():
                metric.attach(trainer, name)

            # Get unit configuration from metrics config
            unit_conversions = None
            unit_labels = None
            if hasattr(metrics_cfg, "unit_conversions"):
                unit_conversions = OmegaConf.to_container(metrics_cfg.unit_conversions)
            if hasattr(metrics_cfg, "unit_labels"):
                unit_labels = OmegaConf.to_container(metrics_cfg.unit_labels)

            # Log validation metrics after validation
            def log_val_metrics(engine):
                metrics_dict = engine.state.metrics
                formatted = format_metrics(metrics_dict, unit_conversions, unit_labels)
                logging.info("Validation Metrics - Epoch %d:\n%s", trainer.state.epoch, formatted)

            validator.add_event_handler(Events.COMPLETED, log_val_metrics)

            # Log training metrics after each epoch
            def log_train_metrics(engine):
                metrics_dict = engine.state.metrics
                formatted = format_metrics(metrics_dict, unit_conversions, unit_labels)
                logging.info("Training Metrics - Epoch %d:\n%s", engine.state.epoch, formatted)

            trainer.add_event_handler(Events.EPOCH_COMPLETED, log_train_metrics)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), validator.run, data=val_loader)

    # Attach scheduler if configured
    if scheduler is not None:
        logging.info("Attaching learning rate scheduler: %s", type(scheduler).__name__)
        validator.add_event_handler(Events.COMPLETED, scheduler)

        # Add early termination based on low LR if configured
        if hasattr(train_cfg.scheduler, 'terminate_on_low_lr'):
            low_lr = train_cfg.scheduler.terminate_on_low_lr
            logging.info("Will terminate training if LR drops below %.2e", low_lr)
            terminator = TerminateOnLowLR(optimizer, low_lr)
            trainer.add_event_handler(Events.EPOCH_STARTED, terminator)

    checkpoint_cfg = getattr(train_cfg, "checkpoint", None)
    if checkpoint_cfg is not None and idist.get_local_rank() == 0:
        from ignite.handlers import ModelCheckpoint, global_step_from_engine

        kwargs = OmegaConf.to_container(checkpoint_cfg.kwargs)
        kwargs["global_step_transform"] = global_step_from_engine(trainer)
        kwargs["dirname"] = checkpoint_cfg.dirname
        kwargs["filename_prefix"] = checkpoint_cfg.filename_prefix
        checkpointer = ModelCheckpoint(**kwargs)

        # Prepare checkpoint objects dict
        to_save = {"model": unwrap_module(model), "optimizer": optimizer}
        if scheduler is not None:
            to_save["scheduler"] = scheduler

        validator.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, to_save)

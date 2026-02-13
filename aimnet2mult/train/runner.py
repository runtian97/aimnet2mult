"""Training orchestration for mixed-fidelity AIMNet2."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from ignite import distributed as idist
from omegaconf import OmegaConf

from ..data import create_mixed_fidelity_loaders
from .engine import create_evaluator, create_trainer

from ..config import build_module  # noqa: E402
from ..modules import Forces  # noqa: E402
from .utils import get_optimizer, get_scheduler, unwrap_module, setup_wandb  # noqa: E402
from ..models.mixed_fidelity_aimnet2 import MixedFidelityAIMNet2  # noqa: E402
from ..models.aimnet2 import AIMNet2  # noqa: E402

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

    # Auto-detect: single fidelity (1 dataset) vs multi-fidelity (>1 datasets)
    num_fidelities = len(train_cfg.data.fidelity_datasets)
    is_single_fidelity = num_fidelities == 1

    if is_single_fidelity:
        logging.info("Single dataset detected - using base AIMNet2 model (aimnet2 compatible)")
        # Build base model directly from config
        model_config = OmegaConf.to_container(model_cfg, resolve=True)
        if isinstance(model_config, dict) and 'kwargs' in model_config:
            cfg = model_config['kwargs']
        else:
            cfg = model_config

        # Build outputs
        outputs_cfg = cfg['outputs']
        outputs = build_module(outputs_cfg)

        model = AIMNet2(
            aev=cfg['aev'],
            nfeature=cfg['nfeature'],
            d2features=cfg['d2features'],
            ncomb_v=cfg['ncomb_v'],
            hidden=cfg['hidden'],
            aim_size=cfg['aim_size'],
            outputs=outputs,
            num_charge_channels=cfg.get('num_charge_channels', 1),
            max_z=cfg.get('max_z', 128),
        )
        logging.info("Base AIMNet2 model built with max_z=%d, num_charge_channels=%d",
                     cfg.get('max_z', 128), cfg.get('num_charge_channels', 1))
    else:
        logging.info("Multiple datasets detected (%d) - using multi-fidelity AIMNet2 model", num_fidelities)
        model = MixedFidelityAIMNet2(
            base_model_config=OmegaConf.to_container(model_cfg, resolve=True),
            num_fidelities=num_fidelities,
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
                elif pretrained_size > current_afv_size:
                    logging.info(f"Truncating afv embedding from {pretrained_size} to {current_afv_size}")
                    state_dict['afv.weight'] = pretrained_afv[:current_afv_size]

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
                        elif pretrained_size > current_size:
                            logging.info(f"Truncating {key} from {pretrained_size} to {current_size}")
                            state_dict[key] = pretrained_shifts[:current_size]

        result = unwrapped.load_state_dict(state_dict, strict=False)
        logging.info(result)

        # Store checkpoint for restoring optimizer/scheduler state later
        _checkpoint_to_restore = checkpoint
    else:
        _checkpoint_to_restore = None

    logging.info("Creating mixed-fidelity data loaders...")
    train_loader, val_loader = create_mixed_fidelity_loaders(train_cfg)

    logging.info("Setting up optimizer...")
    optimizer = get_optimizer(model, train_cfg.optimizer)
    if torch.cuda.device_count() > 0:
        optimizer = idist.auto_optim(optimizer)

    scheduler = None
    if train_cfg.scheduler is not None:
        scheduler = get_scheduler(optimizer, train_cfg.scheduler)

    # Restore optimizer and scheduler state if continuing from checkpoint
    if _checkpoint_to_restore is not None:
        if isinstance(_checkpoint_to_restore, dict):
            if "optimizer" in _checkpoint_to_restore:
                try:
                    optimizer.load_state_dict(_checkpoint_to_restore["optimizer"])
                    logging.info("Restored optimizer state from checkpoint")
                except Exception as e:
                    logging.warning(f"Could not restore optimizer state: {e}")

            if scheduler is not None and "scheduler" in _checkpoint_to_restore:
                try:
                    scheduler.load_state_dict(_checkpoint_to_restore["scheduler"])
                    logging.info("Restored scheduler state from checkpoint")
                except Exception as e:
                    logging.warning(f"Could not restore scheduler state: {e}")

    logging.info("Creating loss function...")
    loss = build_module(OmegaConf.to_container(train_cfg.loss, resolve=True))

    logging.info("Building training engine...")
    trainer = create_trainer(model, optimizer, loss, device, use_base_model=is_single_fidelity)
    validator = create_evaluator(model, device, use_base_model=is_single_fidelity)

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

            # Instantiate metrics for validator (same as aimnet2: attach with name 'multi')
            val_metric = metric_class(**kwargs)
            val_metric.attach_loss(loss_fn)
            val_metric.attach(validator, 'multi')

            # Instantiate separate metrics for trainer
            # Note: trainer returns loss.item() for WandB, so we use custom handler
            # to extract pred/y from engine.state
            train_metric = metric_class(**kwargs)
            train_metric.attach_loss(loss_fn)

            # Manually attach with custom output extraction from engine.state
            @trainer.on(Events.ITERATION_COMPLETED)
            def _update_train_metric(engine):
                if hasattr(engine.state, 'last_pred') and hasattr(engine.state, 'last_y'):
                    train_metric.update((engine.state.last_pred, engine.state.last_y))

            @trainer.on(Events.EPOCH_STARTED)
            def _reset_train_metric(engine):
                train_metric.reset()

            @trainer.on(Events.EPOCH_COMPLETED)
            def _compute_train_metric(engine):
                result = train_metric.compute()
                if not hasattr(engine.state, 'metrics'):
                    engine.state.metrics = {}
                engine.state.metrics.update(result)

            # Log validation metrics after validation
            # Note: ignite merges dict returns from compute() directly into state.metrics
            def log_val_metrics(engine):
                metrics_dict = engine.state.metrics
                if metrics_dict:
                    # Filter out internal keys and format nicely
                    filtered = {k: v for k, v in metrics_dict.items() if not k.startswith('_')}
                    logging.info("Validation Metrics - Epoch %d:\n%s", trainer.state.epoch, str(filtered))

            validator.add_event_handler(Events.COMPLETED, log_val_metrics)

            # Log training metrics after each epoch
            def log_train_metrics(engine):
                metrics_dict = engine.state.metrics
                if metrics_dict:
                    filtered = {k: v for k, v in metrics_dict.items() if not k.startswith('_')}
                    logging.info("Training Metrics - Epoch %d:\n%s", engine.state.epoch, str(filtered))

            trainer.add_event_handler(Events.EPOCH_COMPLETED, log_train_metrics)

        elif metrics_cfg.get("enabled", True):
            # Legacy interface (backward compatibility)
            properties = train_cfg.data.y if hasattr(train_cfg.data, "y") else []

            # Create metrics for validator
            val_metrics = create_metrics(loss_fn, properties, device="cpu")
            for name, metric in val_metrics.items():
                metric.attach(validator, name)

            # Create separate metrics for trainer
            # Note: trainer returns loss.item() for WandB, so we use custom handlers
            train_metrics = create_metrics(loss_fn, properties, device="cpu")

            @trainer.on(Events.ITERATION_COMPLETED)
            def _update_legacy_train_metrics(engine):
                if hasattr(engine.state, 'last_pred') and hasattr(engine.state, 'last_y'):
                    output = (engine.state.last_pred, engine.state.last_y)
                    for metric in train_metrics.values():
                        metric.update(output)

            @trainer.on(Events.EPOCH_STARTED)
            def _reset_legacy_train_metrics(engine):
                for metric in train_metrics.values():
                    metric.reset()

            @trainer.on(Events.EPOCH_COMPLETED)
            def _compute_legacy_train_metrics(engine):
                if not hasattr(engine.state, 'metrics'):
                    engine.state.metrics = {}
                for name, metric in train_metrics.items():
                    engine.state.metrics[name] = metric.compute()

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

    # Run validation - configurable frequency
    # New format: log_frequency.val (iterations)
    # Old format: val_frequency.mode + val_frequency.every (backward compatible)
    log_frequency = train_cfg.get("log_frequency", None)
    val_frequency = train_cfg.get("val_frequency", None)

    if log_frequency is not None and log_frequency.get("val") is not None:
        # New format: log_frequency.val = number of iterations
        val_every = log_frequency.get("val", 10)
        logging.info("Validation will run every %d iterations (log_frequency.val)", val_every)
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=val_every), validator.run, data=val_loader)
    elif val_frequency is not None:
        # Old format: val_frequency with mode
        if val_frequency.get("mode") == "iterations":
            val_every = val_frequency.get("every", 200)
            logging.info("Validation will run every %d iterations (val_frequency)", val_every)
            trainer.add_event_handler(Events.ITERATION_COMPLETED(every=val_every), validator.run, data=val_loader)
        elif val_frequency.get("mode") == "epochs":
            val_every = val_frequency.get("every", 1)
            logging.info("Validation will run every %d epochs", val_every)
            trainer.add_event_handler(Events.EPOCH_COMPLETED(every=val_every), validator.run, data=val_loader)
    else:
        # Default: run validation once per epoch (backward compatible)
        logging.info("Validation will run once per epoch (default)")
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), validator.run, data=val_loader)

    # Copy validation metrics to trainer for scheduler access
    # This allows metric-based schedulers (e.g., ReduceLROnPlateau) to use validation loss
    @validator.on(Events.COMPLETED)
    def _copy_val_metrics_to_trainer(engine):
        if hasattr(engine.state, 'metrics') and engine.state.metrics:
            if not hasattr(trainer.state, 'metrics'):
                trainer.state.metrics = {}
            # Copy validation loss as 'val_loss' for scheduler
            if 'loss' in engine.state.metrics:
                trainer.state.metrics['val_loss'] = engine.state.metrics['loss']
            # Also copy all validation metrics with 'val_' prefix for reference
            for key, value in engine.state.metrics.items():
                trainer.state.metrics[f'val_{key}'] = value

    # Attach scheduler if configured
    if scheduler is not None:
        logging.info("Attaching learning rate scheduler: %s", type(scheduler).__name__)

        # Determine scheduler type and attach appropriately
        scheduler_class_name = type(scheduler).__name__
        is_metric_based = 'ReduceLROnPlateau' in scheduler_class_name
        is_ignite_param_scheduler = hasattr(scheduler, 'optimizer') and hasattr(scheduler, 'param_name')

        if is_metric_based:
            # Metric-based schedulers step after validation completes
            logging.info("Metric-based scheduler detected - will step after validation using validation loss")
            # Step scheduler after validation completes (when val_loss is available)
            @validator.on(Events.COMPLETED)
            def _step_scheduler(engine):
                scheduler(trainer)  # Scheduler reads from trainer.state.metrics['val_loss']
        elif is_ignite_param_scheduler:
            # Ignite param schedulers (e.g., CosineAnnealingScheduler) are called as functions
            logging.info("Ignite param scheduler detected - will step every iteration")
            trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
        else:
            # PyTorch LR schedulers step once per epoch
            logging.info("PyTorch LR scheduler detected - will step once per epoch")
            trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

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

        # Save best models based on validation loss (lower is better)
        save_best = checkpoint_cfg.get("save_best", False)
        if save_best:
            # score_function returns higher value for better models
            # Since we minimize loss, we return negative loss
            score_function = lambda engine: -engine.state.metrics.get('loss', float('inf'))
            kwargs["score_function"] = score_function
            kwargs["score_name"] = "val_loss"
            logging.info("Checkpointing: saving best %d models by validation loss", kwargs.get("n_saved", 1))
        else:
            logging.info("Checkpointing: saving most recent %d models", kwargs.get("n_saved", 1))

        checkpointer = ModelCheckpoint(**kwargs)

        # Prepare checkpoint objects dict - only save model weights
        to_save = {"model": unwrap_module(model)}

        validator.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, to_save)

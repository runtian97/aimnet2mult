import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import omegaconf
from omegaconf import OmegaConf
import torch
from torch import nn, Tensor
from ignite import distributed as idist
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, ProgressBar, TerminateOnNan, global_step_from_engine
from ..config import build_module, get_init_module, get_module, load_yaml
from ..data.sgdataset import SizeGroupedDataset
from ..modules import Forces


def load_dataset(cfg: omegaconf.DictConfig, kind='train'):
    # only load required subset of keys  
    keys = list(cfg.x) + list(cfg.y)
    # in DDP setting, will only load 1/WORLD_SIZE of the data
    if idist.get_world_size() > 1 and not cfg.ddp_load_full_dataset:
        shard = (idist.get_local_rank(), idist.get_world_size())
    else:
        shard = None
    
    extra_kwargs = {
            'keys': keys,
            'shard': shard,
        }
    cfg.datasets[kind].kwargs.update(extra_kwargs)
    cfg.datasets[kind].args = [cfg[kind]]
    ds = build_module(OmegaConf.to_container(cfg.datasets[kind]))
    ds = apply_sae(ds, cfg)
    return ds


def apply_sae(ds: SizeGroupedDataset, cfg: omegaconf.DictConfig):
    for k, c in cfg.sae.items():
        if c is not None and k in cfg.y:
            sae = load_yaml(c.file)
            unique_numbers = set(np.unique(ds.concatenate('numbers').tolist()))
            assert set(sae.keys()).issubset(unique_numbers), f'Keys in SAE file {c.file} do not cover all the dataset atoms'
            if c.mode == 'linreg':
                ds.apply_peratom_shift(k, k, sap_dict=sae)
            elif c.mode == 'logratio':
                ds.apply_pertype_logratio(k, k, sap_dict=sae)
            else:
                raise ValueError(f'Unknown SAE mode {c.mode}')
            for g in ds.groups:
                g[k] = g[k].astype('float32')
    return ds


def get_optimizer(model: nn.Module, cfg: omegaconf.DictConfig):
    logging.info(f'Building optimizer')
    param_groups = dict()
    for k, c in cfg.param_groups.items():
        c = OmegaConf.to_container(c)
        c.pop('re')
        param_groups[k] = {'params': [], **c}
    param_groups['default'] = {'params': []}
    logging.info(f'Default parameters: {cfg.kwargs}')
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        _matched = False
        for k, c in cfg.param_groups.items():
            if re.search(c.re, n):
                param_groups[k]['params'].append(p)
                logging.info(f'{n}: {c}')
                _matched = True
                break
        if not _matched:
            param_groups['default']['params'].append(p)
    d = OmegaConf.to_container(cfg)
    d['args'] = [[v for v in param_groups.values() if len(v['params'])]]
    optimizer = get_init_module(d['class'], d['args'], d['kwargs'])
    logging.info(f'Optimizer: {optimizer}')
    logging.info(f'Trainable parameters:')
    N = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            logging.info(f'{n}: {p.shape}')
        N += p.numel()
    logging.info(f'Total number of trainable parameters: {N}')
    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, cfg: omegaconf.DictConfig):
    d = OmegaConf.to_container(cfg)
    d['args'] = [optimizer]
    scheduler = build_module(d)
    return scheduler


def get_loss(cfg: omegaconf.DictConfig):
    loss = build_module(OmegaConf.to_container(cfg))
    return loss


def set_trainable_parameters(model: nn.Module, force_train: List[str], force_no_train: List[str]) -> nn.Module:
    for n, p in model.named_parameters():
        if any(re.search(x, n) for x in force_no_train):
            p.requires_grad_(False)
        if any(re.search(x, n) for x in force_train):
            p.requires_grad_(True)
    return model


def unwrap_module(net):
    if isinstance(net, Forces):
        net = net.module
        return unwrap_module(net)
    elif isinstance(net, torch.nn.parallel.DistributedDataParallel):
        net = net.module
        return unwrap_module(net)
    else:
        return net


def prepare_batch(batch: Dict[str, Tensor], device='cuda', non_blocking=True) -> Dict[str, Tensor]:
    for k, v in batch.items():
        batch[k] = v.to(device, non_blocking=non_blocking)
    return batch


def setup_wandb(cfg, model_cfg, model, trainer, validator, optimizer):
    import wandb
    from ignite.handlers import WandBLogger, global_step_from_engine
    from ignite.handlers.wandb_logger import OptimizerParamsHandler

    init_kwargs = OmegaConf.to_container(cfg.wandb.init, resolve=True)
    wandb.init(**init_kwargs)
    wandb_logger = WandBLogger(init=False)

    OmegaConf.save(model_cfg, wandb.run.dir + '/model.yaml')
    OmegaConf.save(cfg, wandb.run.dir + '/train.yaml')

    # Log training loss every 200 iterations
    wandb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=200),
        output_transform=lambda output: {"loss": trainer.state.loss},
        tag='train'
        )

    # Log batch-level RMSE every 200 iterations for high-frequency monitoring
    def log_batch_rmse(engine):
        if hasattr(engine.state, 'batch_rmse') and engine.state.batch_rmse:
            # Apply unit conversions (eV to kcal/mol)
            metrics = {}
            ev_to_kcal = 23.06

            if 'energy' in engine.state.batch_rmse:
                metrics['train/E_rmse'] = engine.state.batch_rmse['energy'] * ev_to_kcal
            if 'forces' in engine.state.batch_rmse:
                metrics['train/F_rmse'] = engine.state.batch_rmse['forces'] * ev_to_kcal
            if 'charges' in engine.state.batch_rmse:
                metrics['train/q_rmse'] = engine.state.batch_rmse['charges']
            if 'spin_charges' in engine.state.batch_rmse:
                metrics['train/s_rmse'] = engine.state.batch_rmse['spin_charges']

            if metrics:
                wandb.log(metrics, step=trainer.state.iteration)

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=200), log_batch_rmse)

    # Log validation metrics - they are stored under engine.state.metrics['metrics']
    # We need to extract and flatten them for wandb logging
    def log_validation_metrics(engine):
        if 'metrics' in engine.state.metrics:
            metrics_dict = engine.state.metrics['metrics']
            # Flatten the nested metrics dictionary for wandb with 'val/' prefix
            flat_metrics = {}
            if isinstance(metrics_dict, dict):
                for key, value in metrics_dict.items():
                    # Convert tensor to scalar if needed
                    if hasattr(value, 'item'):
                        value = value.item()
                    if isinstance(value, (int, float)):
                        flat_metrics[f'val/{key}'] = value
            if flat_metrics:
                wandb.log(flat_metrics, step=trainer.state.iteration)

    validator.add_event_handler(Events.EPOCH_COMPLETED, log_validation_metrics)

    # Log training metrics at end of each epoch (if metrics are enabled)
    def log_training_metrics(engine):
        if 'metrics' in engine.state.metrics:
            metrics_dict = engine.state.metrics['metrics']
            flat_metrics = {}
            if isinstance(metrics_dict, dict):
                for key, value in metrics_dict.items():
                    # Convert tensor to scalar if needed
                    if hasattr(value, 'item'):
                        value = value.item()
                    if isinstance(value, (int, float)):
                        flat_metrics[f'train_epoch/{key}'] = value
            if flat_metrics:
                wandb.log(flat_metrics, step=trainer.state.iteration)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_metrics)

    class EpochLRLogger(OptimizerParamsHandler):
        def __call__(self, engine, logger, event_name):
            global_step = engine.state.iteration
            params = {
                '{}_{}'.format(self.param_name, i): float(g[self.param_name])
                for i, g in enumerate(self.optimizer.param_groups)
            }
            logger.log(params, step=global_step, sync=self.sync)

    wandb_logger.attach(
        trainer,
        log_handler=EpochLRLogger(optimizer),
        event_name=Events.EPOCH_STARTED
        )
    
    score_function = lambda engine: 1.0 / engine.state.metrics['loss']
    model_checkpoint = ModelCheckpoint(
            wandb.run.dir, n_saved=1, filename_prefix='best',
            require_empty=False, score_function=score_function,
            global_step_transform=global_step_from_engine(trainer)
        )
    validator.add_event_handler(Events.EPOCH_COMPLETED, model_checkpoint, {'model': unwrap_module(model)})

    if cfg.wandb.watch_model is not None:
        wandb.watch(unwrap_module(model), **OmegaConf.to_container(cfg.wandb.watch_model, resolve=True))

# Learning Rate Scheduler Guide

## Philosophy: Why Ignite Schedulers?

This package **recommends using Ignite schedulers** over PyTorch schedulers because:

1. **Iteration-level control**: Update LR every iteration (not per epoch) → smoother LR trajectories
2. **Fine-grained schedules**: Precise control over warmup, cycles, and decay patterns
3. **Better for variable epoch lengths**: When `batches_per_epoch` varies or is truncated
4. **Consistent with training loop**: Ignite engine already manages the training loop

## Recent Changes

The scheduler implementation has been updated to:

1. **Support Ignite param schedulers**: Automatically detects and configures ignite schedulers (adds `optimizer` and `param_name` to kwargs)

2. **Use validation loss for metric-based schedulers**: ReduceLROnPlateau now uses validation loss instead of training loss to prevent overfitting

3. **Smart scheduler detection**: The code automatically detects scheduler type:
   - **Metric-based** (ReduceLROnPlateau): Steps after validation completes
   - **Ignite param schedulers** (CosineAnnealing, Linear, etc.): Step every iteration
   - **PyTorch LR schedulers**: Step once per epoch

4. **Proper stepping behavior**:
   - **ReduceLROnPlateau**: After validation (when val_loss is available)
   - **Ignite schedulers**: Every iteration (smoother curves)
   - **PyTorch schedulers**: Once per epoch (coarser updates)

## Quick Start

### Using Ignite CosineAnnealingScheduler (Recommended for most training)

In your `train.yaml`:

```yaml
scheduler:
  class: ignite.handlers.param_scheduler.CosineAnnealingScheduler
  kwargs:
    # param_name and optimizer are added automatically
    start_value: 1.0e-4        # Starting learning rate
    end_value: 1.0e-6          # Minimum learning rate at end of cycle
    cycle_size: 200000         # Number of iterations per cycle
    cycle_mult: 1.0            # Multiply cycle_size after each cycle
    start_value_mult: 1.0      # Multiply start_value after each cycle
    end_value_mult: 1.0        # Multiply end_value after each cycle
  terminate_on_low_lr: 0.0
```

**When it steps**: Every training iteration (smooth LR curve)

**Calculating cycle_size**:
```
cycle_size = batches_per_epoch × num_epochs_per_cycle

Example:
  batches_per_epoch: 20000 (from train.yaml)
  Want one cycle over 10 epochs: 20000 × 10 = 200000
  Want one cycle over full 200 epochs: 20000 × 200 = 4000000
```

See `examples/config/train_cosine.yaml` for a complete working example.

### Using ReduceLROnPlateau (Adaptive, no manual tuning)

In your `train.yaml`:

```yaml
scheduler:
  class: ignite.handlers.param_scheduler.ReduceLROnPlateauScheduler
  kwargs:
    metric_name: val_loss  # Uses validation loss
    factor: 0.9            # Multiply LR by 0.9 when reducing
    patience: 2            # Wait 2 validations before reducing
  terminate_on_low_lr: 1.0e-7
```

**When it steps**: After each validation run (frequency controlled by `log_frequency.val`)

### Using Cosine with Warm Restarts (Ignite)

In your `train.yaml`:

```yaml
scheduler:
  class: ignite.handlers.param_scheduler.CosineAnnealingScheduler
  kwargs:
    start_value: 1.0e-4
    end_value: 1.0e-6
    cycle_size: 40000          # First cycle: 2 epochs × 20000 iters
    cycle_mult: 2.0            # Double cycle length after each restart
    start_value_mult: 1.0      # Keep peak LR constant (or use 0.95 to decay)
    end_value_mult: 1.0
  terminate_on_low_lr: 0.0
```

**When it steps**: Every training iteration

**Restart schedule** (with cycle_size=40000, cycle_mult=2.0):
- Cycle 1: iterations 0-39999 (40k iters = 2 epochs)
- Cycle 2: iterations 40000-119999 (80k iters = 4 epochs)
- Cycle 3: iterations 120000-279999 (160k iters = 8 epochs)

### Using Linear Warmup + Decay (Ignite)

In your `train.yaml`:

```yaml
scheduler:
  class: ignite.handlers.param_scheduler.LinearCyclicalScheduler
  kwargs:
    start_value: 1.0e-6        # Start from low LR
    end_value: 1.0e-4          # Warmup to this value
    cycle_size: 100000         # 50k warmup + 50k decay
  terminate_on_low_lr: 0.0
```

**When it steps**: Every training iteration

## How Scheduler Type is Detected

The code automatically detects scheduler type by class name:

```python
is_metric_based = 'ReduceLROnPlateau' in scheduler_class_name
```

- If metric-based: Attaches to `validator` completion event
- If step-based: Attaches to `trainer` epoch completion event

## Interaction with Validation Frequency

### With ReduceLROnPlateau

```yaml
wandb:
  log_frequency:
    train: 500
    val: 500       # Validate every 500 iterations

scheduler:
  class: ignite.handlers.param_scheduler.ReduceLROnPlateauScheduler
  kwargs:
    metric_name: val_loss
    patience: 2    # Wait 2 VALIDATIONS (not epochs)
```

- Validation runs every 500 training iterations
- Scheduler checks validation loss after each validation
- With `patience: 2`, LR reduces if no improvement for 2 consecutive validations
- This is more iterations than 2 epochs (e.g., 2 × 500 = 1000 iterations)

### With Cosine Schedulers

```yaml
scheduler:
  class: torch.optim.lr_scheduler.CosineAnnealingLR
  kwargs:
    T_max: 100
```

- Steps once per **epoch** regardless of validation frequency
- Validation frequency only affects monitoring, not LR schedule
- Independent of `log_frequency.val` setting

## Available Metrics for ReduceLROnPlateau

After validation completes, these metrics are available:

- `val_loss` - Total validation loss (recommended)
- `val_energy_loss` - Energy component loss
- `val_forces_loss` - Forces component loss
- `val_E_mae` - Energy MAE
- `val_E_rmse` - Energy RMSE
- `val_F_mae` - Forces MAE
- `val_F_rmse` - Forces RMSE
- etc.

Example using RMSE instead of loss:

```yaml
scheduler:
  kwargs:
    metric_name: val_E_rmse  # Reduce LR based on energy RMSE
```

## Complete Examples

### Example 1: Standard Training

```yaml
# train.yaml
trainer:
  epochs: 200

wandb:
  log_frequency:
    train: 200
    val: 1000      # Validate every 1000 iterations

scheduler:
  class: ignite.handlers.param_scheduler.ReduceLROnPlateauScheduler
  kwargs:
    metric_name: val_loss
    factor: 0.9
    patience: 3     # Reduce after 3 validations without improvement
  terminate_on_low_lr: 1.0e-7
```

### Example 2: Fast Training with Cosine Schedule

```yaml
# train.yaml
trainer:
  epochs: 50

wandb:
  log_frequency:
    train: 100
    val: 500

scheduler:
  class: torch.optim.lr_scheduler.CosineAnnealingLR
  kwargs:
    T_max: 25       # 50 epochs / 2
    eta_min: 1.0e-7
```

### Example 3: Cosine with Restarts

```yaml
# train.yaml
trainer:
  epochs: 150

scheduler:
  class: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
  kwargs:
    T_0: 10        # First cycle: 10 epochs
    T_mult: 2      # Each cycle 2x longer
    eta_min: 1.0e-7

# Restart schedule:
# Cycle 1: epochs 0-9    (10 epochs)
# Cycle 2: epochs 10-29  (20 epochs)
# Cycle 3: epochs 30-69  (40 epochs)
# Cycle 4: epochs 70-149 (80 epochs)
```

## Common Issues

### Issue: Scheduler not reducing LR

**Possible causes**:

1. **Using training loss instead of validation loss**:
   ```yaml
   # Wrong (uses training loss)
   metric_name: loss

   # Correct (uses validation loss)
   metric_name: val_loss
   ```

2. **Patience too high**: Increase validation frequency or decrease patience
   ```yaml
   # Validate more frequently
   log_frequency:
     val: 500  # Was 5000

   # Or reduce patience
   scheduler:
     kwargs:
       patience: 2  # Was 5
   ```

3. **Min delta too large**: Validation loss is improving but below threshold
   ```yaml
   scheduler:
     kwargs:
       min_delta: 0.0  # Set to 0 or very small value
   ```

### Issue: Cosine schedule completes before training ends

```yaml
# If training for 200 epochs but cosine finishes at epoch 100
scheduler:
  class: torch.optim.lr_scheduler.CosineAnnealingLR
  kwargs:
    T_max: 200  # Increase to match total epochs
```

### Issue: Training stops early with cosine scheduler

```yaml
# Remove terminate_on_low_lr for cosine schedulers
scheduler:
  class: torch.optim.lr_scheduler.CosineAnnealingLR
  kwargs:
    T_max: 100
  terminate_on_low_lr: null  # Add this
```

## Monitoring

You can monitor scheduler behavior in WandB:

1. **Learning rate**: Logged automatically as `lr_0`, `lr_1`, etc.
2. **Validation loss**: Logged as `val/loss`
3. **Validation metrics**: Logged as `val/E_mae`, `val/F_rmse`, etc.

Look for:
- LR reductions coinciding with validation loss plateaus
- Smooth LR decay for cosine schedulers
- Periodic restarts for CosineAnnealingWarmRestarts

## Ignite vs PyTorch Schedulers

### When Each Type Steps

| Scheduler Type | Steps When | Granularity | Example |
|----------------|------------|-------------|---------|
| **Ignite Param Schedulers** | Every iteration | Fine (per batch) | CosineAnnealingScheduler |
| **PyTorch LR Schedulers** | Every epoch | Coarse (per epoch) | CosineAnnealingLR |
| **Metric-based** | After validation | Variable | ReduceLROnPlateau |

### Ignite Schedulers (Recommended)

**Pros:**
- Update every iteration → smoother LR curves
- More precise control over schedule
- Can implement warmup, multi-phase schedules easily
- Better for variable-length epochs

**Cons:**
- Need to calculate `cycle_size` in iterations
- Less familiar to users coming from PyTorch

**Available Ignite schedulers:**
- `CosineAnnealingScheduler` - Smooth cosine decay
- `LinearCyclicalScheduler` - Linear warmup/decay
- `ConcatScheduler` - Combine multiple phases
- `ReduceLROnPlateauScheduler` - Metric-based

**Configuration pattern:**
```yaml
scheduler:
  class: ignite.handlers.param_scheduler.CosineAnnealingScheduler
  kwargs:
    # optimizer and param_name added automatically
    start_value: 1.0e-4
    end_value: 1.0e-6
    cycle_size: 200000  # In iterations, not epochs!
```

### PyTorch Schedulers (For Compatibility)

**Pros:**
- Familiar API for PyTorch users
- Work with epochs (easier to reason about)
- Standard in most PyTorch codebases

**Cons:**
- Coarse updates (once per epoch only)
- Less smooth LR curves
- Harder to implement complex schedules

**Available PyTorch schedulers:**
- `torch.optim.lr_scheduler.CosineAnnealingLR`
- `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`
- `torch.optim.lr_scheduler.StepLR`
- `torch.optim.lr_scheduler.OneCycleLR`
- etc.

**Configuration pattern:**
```yaml
scheduler:
  class: torch.optim.lr_scheduler.CosineAnnealingLR
  kwargs:
    T_max: 100        # In epochs
    eta_min: 1.0e-7
```

### Recommendation

**Use Ignite schedulers** for this package because:
1. Better integration with the Ignite training loop
2. Smoother LR schedules (iteration-level updates)
3. More flexible for complex training regimes
4. Consistent with the rest of the training infrastructure

**Use PyTorch schedulers** only if:
- You have an existing PyTorch training script to port
- You need a scheduler not available in Ignite
- Epoch-level updates are sufficient for your use case

## Calculating Cycle Size for Ignite Schedulers

This is the most common source of confusion. Remember:

```
cycle_size = batches_per_epoch × num_epochs

Where batches_per_epoch comes from your train.yaml:
  data.samplers.train.kwargs.batches_per_epoch
```

**Examples** (with batches_per_epoch=20000):

```yaml
# Full training run (200 epochs)
cycle_size: 4000000  # 20000 × 200

# Half training run (100 epochs)
cycle_size: 2000000  # 20000 × 100

# 10 epoch cycle
cycle_size: 200000   # 20000 × 10

# Single epoch
cycle_size: 20000    # 20000 × 1
```

**Pro tip**: For warm restarts, start with a smaller `cycle_size` (e.g., 2-5 epochs worth) and use `cycle_mult > 1.0` to increase cycle length over time.

## See Also

- `examples/config/scheduler_examples.yaml` - Comprehensive scheduler examples
- `examples/config/train.yaml` - Default training configuration (ReduceLROnPlateau)
- `examples/config/train_cosine.yaml` - Example with Ignite CosineAnnealingScheduler

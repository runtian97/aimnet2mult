# WandB Logging Fix - Changelog

## Date: 2026-01-16

### Summary
Fixed WandB logging to properly track training loss, validation metrics (RMSE, MAE, R²), and training epoch metrics during mixed-fidelity AIMNet2 training.

### Files Modified

#### 1. `aimnet2mult/train/engine.py`
**Problem:** Training engine was returning `(pred, y)` tuple, causing WandB to fail logging training loss with a type error warning.

**Solution:** Store loss value separately in `engine.state.loss` for WandB logging while still returning `(pred, y)` for metrics computation.

```python
# Line 39-43
# Store loss separately for WandB logging
engine.state.loss = total_loss.item()

# Return pred and y for metrics computation
return pred, y
```

#### 2. `aimnet2mult/train/utils.py`
**Problem:**
- Training loss output_transform was trying to extract from tuple
- Validation metrics were using `wandb_logger.log()` which doesn't work correctly
- Tensor values weren't being converted to scalars

**Solutions:**

a) Fixed training loss handler (line 143):
```python
output_transform=lambda output: {"loss": trainer.state.loss},
```

b) Fixed validation metrics logging (line 149-162):
- Changed from `wandb_logger.log()` to `wandb.log()`
- Added proper tensor-to-scalar conversion with `value.item()`
- Added `val/` prefix for organization

c) Added training epoch metrics logging (line 167-181):
- Logs full-epoch training metrics with `train_epoch/` prefix
- Proper tensor-to-scalar conversion
- Uses `wandb.log()` instead of `wandb_logger.log()`

d) Removed deprecated `sync` parameter from wandb.log() calls

#### 3. `.gitignore` (New file)
Added comprehensive .gitignore to prevent committing:
- Python artifacts (__pycache__, *.pyc, etc.)
- Training outputs (checkpoints/, wandb/, *.pt files)
- IDE files (.vscode/, .idea/)
- Temporary files (*.log, temp/)

### Metrics Now Logged to WandB

#### Training Metrics
- **Every 200 iterations:** `train/loss`
- **End of each epoch:**
  - `train_epoch/E_mae`, `train_epoch/E_rmse`, `train_epoch/E_r2` (Energy)
  - `train_epoch/F_mae`, `train_epoch/F_rmse`, `train_epoch/F_r2` (Forces)
  - `train_epoch/q_mae`, `train_epoch/q_rmse`, `train_epoch/q_r2` (Charges)
  - `train_epoch/s_mae`, `train_epoch/s_rmse`, `train_epoch/s_r2` (Spin charges)
  - `train_epoch/loss`, component losses

#### Validation Metrics (after each epoch)
- `val/E_mae`, `val/E_rmse`, `val/E_r2` (Energy in kcal/mol)
- `val/F_mae`, `val/F_rmse`, `val/F_r2` (Forces in kcal/mol/Å)
- `val/q_mae`, `val/q_rmse`, `val/q_r2` (Charges)
- `val/s_mae`, `val/s_rmse`, `val/s_r2` (Spin charges)
- `val/loss`, `val/energy_loss`, `val/forces_loss`, etc.

### Testing
- Verified with 2-epoch test run on fake dataset
- Confirmed metrics appear in WandB dashboard
- Test run: https://wandb.ai/runtiangao-carnegie-mellon-university/test_project/runs/9p0kz694

### Notes
- Validation metrics appear after each epoch completes
- Training loss appears every 200 iterations
- All metrics properly organized with prefixes for easy dashboard filtering
- Energy/Forces metrics automatically converted to kcal/mol units via scale factors in config

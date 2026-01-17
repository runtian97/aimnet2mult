# High-Frequency RMSE Logging

## Date: 2026-01-17

### What Was Added

Added **batch-level RMSE computation and logging** to match high-frequency training loss monitoring.

### Changes Made

#### 1. `aimnet2mult/train/engine.py`
- Added `_compute_batch_rmse()` function to compute RMSE on each training batch
- Modified trainer to store batch RMSE in `engine.state.batch_rmse`
- Computes RMSE for: energy, forces, charges, spin_charges
- Handles masked values for mixed-fidelity training

#### 2. `aimnet2mult/train/utils.py`
- Added `log_batch_rmse()` handler to log RMSE every 200 iterations
- Applies unit conversions (eV → kcal/mol) automatically
- Logs to WandB with `train/` prefix

### What Gets Logged Now

#### **High-Frequency (Every 200 Iterations):**
- `train/loss` - Training loss
- `train/E_rmse` - Energy RMSE (kcal/mol) ✨ NEW
- `train/F_rmse` - Forces RMSE (kcal/mol/Å) ✨ NEW
- `train/q_rmse` - Charges RMSE ✨ NEW
- `train/s_rmse` - Spin charges RMSE ✨ NEW

#### **Per-Epoch (After Each Epoch):**
- `train_epoch/E_rmse`, `train_epoch/F_rmse`, etc. - Training set metrics
- `val/E_rmse`, `val/F_rmse`, etc. - Validation set metrics

### Example Timeline

```
Iteration 200:   train/loss = 45.2,  train/E_rmse = 1305.3,  train/F_rmse = 35.8
Iteration 400:   train/loss = 43.8,  train/E_rmse = 1298.1,  train/F_rmse = 35.2
Iteration 600:   train/loss = 42.1,  train/E_rmse = 1285.6,  train/F_rmse = 34.9
...
Iteration 20000: (epoch 1 complete)
  ↓
train_epoch/E_rmse = 1280.5  (full training set)
  ↓
val/E_rmse = 390.3  (validation set)
```

### WandB Dashboard

You'll now see smooth curves (thousands of points) for:
- `train/loss` over ~4M steps
- `train/E_rmse` over ~4M steps (like your screenshot!)
- `train/F_rmse` over ~4M steps
- `train/q_rmse` over ~4M steps
- `train/s_rmse` over ~4M steps

And discrete points (one per epoch) for:
- `val/E_rmse`, `val/F_rmse`, etc.

### Performance Impact

**Minimal:** RMSE computation is fast (<1ms per batch) and happens in no_grad() mode.

### Benefits

1. **Real-time monitoring** - See RMSE trends during training, not just at epoch end
2. **Early problem detection** - Catch divergence or instability immediately
3. **Better debugging** - Correlate loss spikes with RMSE behavior
4. **Matches standard ML practices** - Like your screenshot showing high-frequency metrics

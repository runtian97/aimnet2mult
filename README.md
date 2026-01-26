# AIMNet2mult - Multi-Fidelity Neural Network Potential

A PyTorch implementation of AIMNet2 with **mixed-fidelity training** and **per-fidelity dispersion** support. Train a single model on datasets from different computational methods simultaneously.

## Key Features

- **Mixed-Fidelity Training**: Single model learns from multiple data sources with different accuracy levels
- **Per-Fidelity Dispersion**: Add different DFT-D3/D4 dispersion corrections per fidelity during compilation
- **Heterogeneous Labels**: Different datasets can have different available properties (energy, forces, charges, etc.)
- **Automatic Masking**: Loss calculation automatically adapts to available labels per sample
- **Transfer Learning**: Load pretrained weights and fine-tune on new datasets
- **TorchScript Export**: Compile models for fast production inference
- **Wandb Integration**: Built-in experiment tracking and visualization

## Installation

### From GitHub (recommended)

```bash
pip install git+https://github.com/runtian97/aimnet2mult.git
```

### From source (for development)

```bash
git clone https://github.com/runtian97/aimnet2mult.git
cd aimnet2mult
pip install -e .
```

### Optional dependencies

```bash
# Dispersion packages for D3/D4 support
pip install dftd3 dftd4

# Wandb for experiment tracking
pip install wandb
wandb login

# Training dependencies
pip install aimnet2mult[train]

# All optional dependencies
pip install aimnet2mult[all]
```

## Quick Start

### Step 1: Calculate SAE for each fidelity dataset

```bash
python -m aimnet2mult.train.calc_sae fid0_dataset.h5 sae_fid0.yaml
python -m aimnet2mult.train.calc_sae fid1_dataset.h5 sae_fid1.yaml
python -m aimnet2mult.train.calc_sae fid2_dataset.h5 sae_fid2.yaml
```

### Step 2: Train the model

```bash
python -m aimnet2mult.train.cli \
    --config train.yaml \
    --model model.yaml \
    --save model.pt \
    data.fidelity_datasets.0="fid0_dataset.h5" \
    data.fidelity_datasets.1="fid1_dataset.h5" \
    data.fidelity_datasets.2="fid2_dataset.h5" \
    data.sae.energy.files.0="sae_fid0.yaml" \
    data.sae.energy.files.1="sae_fid1.yaml" \
    data.sae.energy.files.2="sae_fid2.yaml"
```

### Step 3: Compile with dispersion

```bash
python -m aimnet2mult.tools.compile_jit \
    --weights model.pt \
    --model model.yaml \
    --output compiled_model \
    --fidelity-level 0 \
    --fidelity-offset 200 \
    --num-fidelities 3 \
    --use-fidelity-readouts true \
    --sae sae_fid0.yaml \
    --dispersion d3bj \
    --dispersion-functional wb97m
```

### Continue Training from Checkpoint

```bash
python -m aimnet2mult.train.cli \
    --config train.yaml \
    --model model.yaml \
    --load pretrained_model.pt \
    --save continued_model.pt \
    ...
```

## Package Structure

```
aimnet2mult/
├── aimnet2mult/                    # Core package
│   ├── __init__.py
│   ├── aev.py                      # Atomic environment vectors
│   ├── calculators.py              # ASE calculator interface
│   ├── constants.py                # Physical constants
│   ├── d3bj_data.pt                # D3-BJ C6 reference coefficients
│   ├── modules.py                  # Neural network modules (MLP, DFTD3, AtomicShift, etc.)
│   ├── nbops.py                    # Neighbor list operations
│   ├── ops.py                      # Tensor operations
│   │
│   ├── config/                     # Configuration system
│   │   ├── __init__.py
│   │   ├── builder.py              # Module instantiation from YAML
│   │   ├── default_model.yaml      # Default model architecture
│   │   └── default_train_mixed_fidelity.yaml  # Default training config
│   │
│   ├── data/                       # Data loading and processing
│   │   ├── __init__.py
│   │   ├── base.py                 # MultiFidelityDataset base class
│   │   ├── collate.py              # Collate function with automatic masking
│   │   ├── loaders.py              # DataLoader creation utilities
│   │   ├── mixed.py                # MixedFidelityDataset (applies atomic number offsets)
│   │   ├── sampler.py              # Mixed-fidelity batch sampler
│   │   └── sgdataset.py            # Size-grouped HDF5 dataset with SAE calculation
│   │
│   ├── models/                     # Neural network architectures
│   │   ├── __init__.py
│   │   ├── aimnet2.py              # Base AIMNet2 model
│   │   ├── base.py                 # Abstract base classes
│   │   ├── jit_wrapper.py          # TorchScript export utilities
│   │   └── mixed_fidelity_aimnet2.py  # Multi-fidelity wrapper
│   │
│   ├── tools/                      # Utilities
│   │   ├── __init__.py
│   │   ├── compile_jit.py          # TorchScript compilation with SAE and dispersion
│   │   ├── create_fake_data.py     # Generate test data
│   │   └── dispersion_params.py    # D3/D4 parameter loading
│   │
│   ├── train/                      # Training infrastructure
│   │   ├── __init__.py
│   │   ├── calc_sae.py             # Self-atomic energy calculation script
│   │   ├── cli.py                  # Command-line interface (--load for transfer learning)
│   │   ├── configuration.py        # Config loading and validation
│   │   ├── engine.py               # Ignite engines (trainer/evaluator)
│   │   ├── fidelity_specific_utils.py  # Fidelity-specific utilities
│   │   ├── loss.py                 # Loss functions with masking support
│   │   ├── metrics.py              # Evaluation metrics
│   │   ├── runner.py               # Training orchestration
│   │   └── utils.py                # Optimizer, scheduler, WandB setup
│   │
│   └── utils/                      # Utility functions
│       ├── __init__.py
│       └── units.py                # Unit conversion utilities
│
├── examples/                       # Example scripts and configs
│   ├── config/
│   │   ├── model.yaml              # Model architecture config
│   │   └── train.yaml              # Training configuration
│   ├── continue_training.sh        # Continual training from checkpoint
│   ├── run_example.sh              # Initial training script
│   └── README.md                   # Example instructions
│
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
└── setup.py
```

## Training Architecture

### Overview

The multi-fidelity AIMNet2 model uses a shared backbone with fidelity-specific readout heads. During training, molecules from different fidelity levels are distinguished by offsetting their atomic numbers.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Multi-Fidelity AIMNet2 Training                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: Molecules from multiple fidelity datasets                           │
│         Fidelity 0: Z → Z        (e.g., wB97M-D3 high-fidelity)            │
│         Fidelity 1: Z → Z + 200  (e.g., B3LYP medium-fidelity)             │
│         Fidelity 2: Z → Z + 400  (e.g., PM7 low-fidelity)                  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Shared Backbone                                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Atomic Feature Vectors (AFV) Embedding                          │  │ │
│  │  │  - Extended embedding table: [0..118, 200..318, 400..518, ...]  │  │ │
│  │  │  - Each fidelity learns its own element representations          │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │                              ↓                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  AEV (Atomic Environment Vectors)                                │  │ │
│  │  │  - Radial symmetry functions                                     │  │ │
│  │  │  - Angular symmetry functions                                    │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │                              ↓                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  Message Passing Layers                                          │  │ │
│  │  │  - Shared convolution weights                                    │  │ │
│  │  │  - Shared MLP transformations                                    │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Fidelity-Specific Readout Heads                                      │ │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                 │ │
│  │  │ Fidelity 0  │   │ Fidelity 1  │   │ Fidelity 2  │                 │ │
│  │  │ - Energy MLP│   │ - Energy MLP│   │ - Energy MLP│                 │ │
│  │  │ - Charges   │   │ - Charges   │   │ - Charges   │                 │ │
│  │  │ - Atomic    │   │ - Atomic    │   │ - Atomic    │                 │ │
│  │  │   Shifts    │   │   Shifts    │   │   Shifts    │                 │ │
│  │  └─────────────┘   └─────────────┘   └─────────────┘                 │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                              │
│  Outputs: energy, forces (via autograd), charges, spin_charges             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Training Concepts

1. **Atomic Number Offsetting**: Each fidelity uses a different atomic number space
   - Fidelity 0: H=1, C=6, N=7, O=8, ...
   - Fidelity 1: H=201, C=206, N=207, O=208, ...
   - Fidelity 2: H=401, C=406, N=407, O=408, ...

2. **Shared vs Fidelity-Specific Parameters**:
   - Shared: Message passing layers, convolutions
   - Per-fidelity: Readout MLPs, atomic shift embeddings

3. **Heterogeneous Label Masking**: Loss automatically masks missing labels per molecule

### Gradient Flow with Heterogeneous Labels

When training with multiple datasets where only some have certain labels (e.g., only Dataset 0 has charges/forces), understanding gradient flow is crucial.

#### Architecture: Shared Backbone + Fidelity-Specific Readouts

```
                    ┌─────────────────────────────────────────────────────────┐
                    │           SHARED MESSAGE PASSING BACKBONE               │
                    │  ┌─────────────────────────────────────────────────┐   │
  Dataset 0 ───────►│  │  Embedding → AEV → MP Iterations → `aim`        │   │
  Dataset 1 ───────►│  │  (All datasets contribute gradients here)       │   │
  Dataset 2 ───────►│  └─────────────────────────────────────────────────┘   │
                    └───────────────────────┬─────────────────────────────────┘
                                            │
                                            ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │         FIDELITY-SPECIFIC READOUT HEADS                 │
                    │  ┌───────────┐  ┌───────────┐  ┌───────────┐           │
                    │  │ Fidelity 0│  │ Fidelity 1│  │ Fidelity 2│           │
                    │  │ energy ✓  │  │ energy ✓  │  │ energy ✓  │           │
                    │  │ forces ✓  │  │ forces ✗  │  │ forces ✗  │           │
                    │  │ charges ✓ │  │ charges ✗ │  │ charges ✗ │           │
                    │  └───────────┘  └───────────┘  └───────────┘           │
                    │  (Only trained    (No gradient   (No gradient          │
                    │   on available     for forces/    for forces/          │
                    │   labels)          charges)       charges)             │
                    └─────────────────────────────────────────────────────────┘
```

#### How Missing Labels Are Masked

**Step 1: Data Collation** (`aimnet2mult/data/collate.py`)

When batching molecules, the collate function creates availability masks:

```python
# From collate.py lines 92-102
for item in dict_list:
    if key in item:
        arr = np.asarray(item[key])
        values.append(arr)
        availability.append(1.0)   # Label present
    else:
        values.append(None)
        availability.append(0.0)   # Label missing - will be masked
```

**Step 2: Loss Computation** (`aimnet2mult/train/loss.py`)

The loss functions check for masks and return zero when labels are missing:

```python
# From loss.py lines 122-136
def mse_loss_fn(y_pred, y_true, key_pred, key_true):
    # If label completely missing from batch, return zero loss
    if key_true not in y_true:
        return torch.zeros((), device=y_pred[key_pred].device, dtype=y_pred[key_pred].dtype)

    x = y_true[key_true]
    y = y_pred[key_pred]
    mask = _get_sample_mask(y_true, key_true, x)

    if mask is None:
        l = torch.nn.functional.mse_loss(x, y)
    else:
        # Per-sample masking: only molecules with labels contribute
        diff2 = (x - y).pow(2).view(x.shape[0], -1).mean(dim=-1)
        l = _apply_sample_mask(diff2, mask)
    return l
```

**Step 3: Masked Averaging** (`aimnet2mult/train/loss.py`)

```python
# From loss.py lines 114-119
def _apply_sample_mask(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.squeeze(-1)
    denom = mask.sum()
    if denom.item() == 0:
        return torch.zeros((), device=values.device, dtype=values.dtype)  # No valid samples
    return (values * mask).sum() / denom  # Average only over valid samples
```

#### Gradient Flow Per Dataset

| Dataset | Has Labels | Contributes to Backbone | Contributes to Readout |
|---------|------------|------------------------|------------------------|
| Dataset 0 | energy, forces, charges | ✓ (via all losses) | ✓ Fidelity 0 heads trained |
| Dataset 1 | energy only | ✓ (via energy loss only) | ✓ Fidelity 1 energy head only |
| Dataset 2 | energy only | ✓ (via energy loss only) | ✓ Fidelity 2 energy head only |

#### Key Insight: Fidelity 0 Benefits Most

**Fidelity 0's model gets the best of both worlds:**

1. **Better backbone representations**: The shared message passing layers see molecules from ALL datasets, learning more diverse atomic environments

2. **Fully trained readout heads**: Since Dataset 0 has all labels (energy, forces, charges, spin_charges), all its readout heads receive gradient updates

3. **Richer training signal**: Forces and charges provide additional supervision that helps the backbone learn better atomic representations

```
Gradient contribution to shared backbone:

  Dataset 0: ═══════════════════════════════════► (energy + forces + charges)
  Dataset 1: ════════════►                        (energy only)
  Dataset 2: ════════════►                        (energy only)
             ─────────────────────────────────────
             MORE molecular diversity, Dataset 0 readouts fully trained
```

**The other fidelities essentially "donate" their molecular diversity to improve the backbone, while only getting energy predictions in return.**

#### Fidelity-Specific Readouts Are Independent

Each fidelity creates its own set of readout heads (`aimnet2mult/models/mixed_fidelity_aimnet2.py`):

```python
# From mixed_fidelity_aimnet2.py lines 95-101
for fid in range(self.num_fidelities):
    # Each fidelity gets INDEPENDENT readout heads
    fid_outputs = build_module(outputs_cfg)  # Creates NEW MLP weights
    ...
    self.fidelity_readouts[str(fid)] = nn.ModuleDict(fid_outputs)
```

This means:
- Fidelity 1's charges head exists but is **never trained** (no gradient signal)
- Fidelity 2's forces head exists but is **never trained** (no gradient signal)
- Only heads with corresponding labels in their dataset receive gradient updates

### SAE + Atomic Number Offsets: Training, Compilation, Inference

- **Training (mixed-fidelity)**:
  - `MixedFidelityDataset` applies atomic number offsets per fidelity level before batching.
  - SAE corrections are loaded per fidelity and applied during data loading (`_apply_sae`), so training targets are residual energies.
  - The model sees offset `numbers` and uses the expanded embedding table (offset blocks).

- **Compilation (per-fidelity JIT)**:
  - `compile_jit.py` slices the trained embedding table and atomic-shift weights for the target fidelity.
  - The compiled model is wrapped with `FidelityModelWithSAE` (and optional D3BJ) to **add SAE back** at inference.
  - SAE lookup uses `shifted_numbers = numbers + fidelity_level * fidelity_offset` internally.

- **Inference (compiled model)**:
  - Provide **unshifted** atomic numbers (standard Z=1..118).
  - The wrapper injects the fidelity label and applies SAE with the appropriate offset.
  - Each compiled `.jpt` is fidelity-specific, so use the matching model for that dataset.

## Complete Training Algorithm Architecture

### Training Loop Overview

The training system is built on **PyTorch Ignite** with three independent subsystems:

```
┌────────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP ARCHITECTURE                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  1. DATA LOADING (Multi-Fidelity Batching)               │ │
│  │     - Load HDF5 datasets per fidelity                    │ │
│  │     - Apply atomic number offsets (Z → Z + fid*offset)   │ │
│  │     - Apply SAE corrections (residual energies)          │ │
│  │     - Create mixed batches with fidelity labels          │ │
│  │     - Apply automatic masking for missing labels         │ │
│  └──────────────────────────────────────────────────────────┘ │
│                           ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  2. FORWARD PASS                                         │ │
│  │     - Embedding lookup (extended table for all fidelities)│
│  │     - AEV computation (radial + angular features)        │ │
│  │     - Message passing (shared backbone)                  │ │
│  │     - Fidelity-specific readouts (energy, charges, etc.) │ │
│  │     - Force computation via autograd                     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                           ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  3. LOSS COMPUTATION (Multi-Task + Masking)              │ │
│  │     - Energy loss (MSE on available samples)             │ │
│  │     - Forces loss (per-atom MSE with masking)            │ │
│  │     - Charges loss (per-atom MSE with masking)           │ │
│  │     - Spin charges loss (per-atom MSE with masking)      │ │
│  │     - Weighted sum → total loss                          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                           ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  4. BACKWARD PASS + OPTIMIZATION                         │ │
│  │     - loss.backward() → compute gradients                │ │
│  │     - Gradient clipping (value=0.4 for stability)        │ │
│  │     - Optimizer step (AdamW with param groups)           │ │
│  │     - Scheduler step (iteration-level LR update)         │ │
│  └──────────────────────────────────────────────────────────┘ │
│                           ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  5. LOGGING & VALIDATION (Configurable Frequency)        │ │
│  │     - Train loss → WandB (every N iterations)            │ │
│  │     - Validation run (every M iterations)                │ │
│  │     - Validation metrics → WandB                         │ │
│  │     - Learning rate → WandB (per epoch)                  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                           ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  6. CHECKPOINTING                                        │ │
│  │     - Save best model by validation loss                 │ │
│  │     - Save optimizer + scheduler state                   │ │
│  │     - Keep top N checkpoints                             │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Three Independent Subsystems

The training loop coordinates three independent subsystems that operate on different schedules:

#### 1. Scheduler (Iteration-Level LR Control)

**Frequency:** Every training iteration

```python
# Ignite param schedulers update every iteration
trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
```

**Supported Schedulers:**
- **Ignite CosineAnnealingScheduler** (Recommended) - Smooth cosine decay per iteration
- **Ignite LinearCyclicalScheduler** - Linear warmup/decay cycles
- **ReduceLROnPlateau** - Metric-based reduction (uses validation loss)
- **PyTorch schedulers** - Standard epoch-based schedulers

**Key Feature:** Iteration-level control for smooth LR curves
```yaml
scheduler:
  class: ignite.handlers.param_scheduler.CosineAnnealingScheduler
  kwargs:
    start_value: 1.0e-4
    end_value: 1.0e-6
    cycle_size: 200000  # In iterations, not epochs!
```

See `examples/config/scheduler_examples.yaml` for 15+ scheduler configurations.

#### 2. WandB Logging

**Frequency:** Configurable via `log_frequency`

```yaml
wandb:
  log_frequency:
    train: 500  # Log training loss every 500 iterations
    val: 500    # Run validation and log metrics every 500 iterations
```

**What Gets Logged:**

Every `log_frequency.train` iterations:
- `train/loss` - Training loss from current batch

Every `log_frequency.val` iterations (when validation completes):
- `val/loss` - Total validation loss
- `val/energy_loss`, `val/forces_loss`, etc. - Component losses
- `val/E_mae`, `val/E_rmse` - Energy metrics (kcal/mol)
- `val/F_mae`, `val/F_rmse` - Forces metrics (kcal/mol/Å)
- `val/q_mae`, `val/q_rmse` - Charges metrics
- `val/s_mae`, `val/s_rmse` - Spin charges metrics

Every epoch start:
- `lr_0`, `lr_1`, ... - Learning rates for all param groups

**Independence:** Logging is completely independent of scheduler and validation frequency.

#### 3. Validation

**Frequency:** Controlled by `log_frequency.val`

```python
# Validation runs every val_every training iterations
trainer.add_event_handler(
    Events.ITERATION_COMPLETED(every=val_every),
    validator.run,
    data=val_loader
)
```

**Process:**
1. Switch model to eval mode
2. Run inference on validation set (no gradients)
3. Compute validation metrics (MAE, RMSE, loss components)
4. Log metrics to WandB
5. Update scheduler (if ReduceLROnPlateau)
6. Save checkpoint (if best validation loss)

### Data Flow: From HDF5 to Predictions

```
HDF5 Datasets (per fidelity)
     ├── fidelity_0.h5 (wB97M-D3, all labels)
     ├── fidelity_1.h5 (B3LYP, energy only)
     └── fidelity_2.h5 (PM7, energy only)
              ↓
┌─────────────────────────────────────┐
│  SizeGroupedDataset                 │
│  - Load by molecule size            │
│  - Apply SAE corrections            │
│  - Store in memory/shard            │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  MixedFidelityDataset               │
│  - Apply atomic number offsets      │
│    Fid 0: Z → Z                     │
│    Fid 1: Z → Z + 200               │
│    Fid 2: Z → Z + 400               │
│  - Attach fidelity labels           │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  MixedFidelitySampler               │
│  - Sample by fidelity weights       │
│  - Create batches (atoms or mols)   │
│  - Shuffle within fidelity          │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Collate Function                   │
│  - Pad to batch size                │
│  - Create availability masks        │
│  - Stack tensors                    │
└─────────────────────────────────────┘
              ↓
        Batch Dict:
        {
          'coord': [B, max_atoms, 3],
          'numbers': [B, max_atoms],  # Offset by fidelity!
          'charge': [B],
          'mult': [B],
          'energy': [B],
          'energy_mask': [B],  # Availability
          'forces': [B, max_atoms, 3],
          'forces_mask': [B],  # May be zeros
          '_fidelities': [B],  # Fidelity labels
        }
              ↓
┌─────────────────────────────────────┐
│  Model Forward Pass                 │
│  1. Embedding: numbers → features   │
│  2. AEV: geometric features         │
│  3. Message passing: atom updates   │
│  4. Fidelity readout: properties    │
└─────────────────────────────────────┘
              ↓
        Predictions:
        {
          'energy': [B],
          'forces': [B, max_atoms, 3],
          'charges': [B, max_atoms],
          'spin_charges': [B, max_atoms],
        }
              ↓
┌─────────────────────────────────────┐
│  Loss Computation                   │
│  - Apply masks to ignore missing    │
│  - Compute per-property losses      │
│  - Weighted sum → total loss        │
└─────────────────────────────────────┘
```

### Training Iteration Timeline

**Example:** 200 epochs × 20,000 batches/epoch = 4,000,000 total iterations

With `log_frequency.train=500`, `log_frequency.val=500`:

```
Iteration    Scheduler    Train Log    Validation    Val Log    Checkpoint
─────────    ─────────    ─────────    ──────────    ───────    ──────────
    1        ✓ step
    2        ✓ step
  ...        ✓ step
  500        ✓ step       ✓ log        ✓ run         ✓ log      ✓ save?
  501        ✓ step
  ...        ✓ step
 1000        ✓ step       ✓ log        ✓ run         ✓ log      ✓ save?
  ...        ✓ step
20000        ✓ step       ✓ log        ✓ run         ✓ log      ✓ save?
           (Epoch 1 complete)
20001        ✓ step
  ...        ✓ step
40000        ✓ step       ✓ log        ✓ run         ✓ log      ✓ save?
           (Epoch 2 complete)
```

**Key Observations:**
- Scheduler steps **4,000,000 times** (every iteration)
- Training logged **8,000 times** (every 500 iterations)
- Validation runs **8,000 times** (every 500 iterations)
- Checkpoints saved **~40 times** (when validation loss improves)

### Optimizer Configuration

Multi-level parameter groups with different learning rates:

```yaml
optimizer:
  class: torch.optim.AdamW
  kwargs:
    lr: 5.0e-5           # Base learning rate
    weight_decay: 1.0e-6
    betas: [0.9, 0.999]
    eps: 1.0e-8

  param_groups:
    embeddings:
      re: 'afv.weight'         # Atomic feature vectors
      lr: 2.5e-5               # Half of base LR
      weight_decay: 0.0        # No regularization

    shifts:
      re: '.*.atomic_shift.shifts.weight'
      weight_decay: 0.0        # No regularization
```

**Rationale:**
- **Lower LR for embeddings:** Prevent catastrophic forgetting of atomic representations
- **No weight decay for embeddings/shifts:** These are lookup tables, not overfit-prone weights
- **Scheduler modulates base LR:** `start_value` and `end_value` in scheduler config

### Metrics Computation

**During Training** (per iteration):
- Loss components stored in `engine.state` for later aggregation

**After Epoch** (training metrics):
- Accumulate predictions and targets over epoch
- Compute MAE, RMSE, R² per property
- Apply unit conversions (eV → kcal/mol)
- Log to console and WandB

**During Validation** (per validation run):
- Run inference on full validation set
- Compute metrics with automatic masking
- Store in `validator.state.metrics`
- Log to WandB with `val/` prefix

**Metric Types:**
```python
# Example metrics output
{
  'loss': 0.0123,           # Total validation loss
  'energy_loss': 0.0100,    # Energy component
  'E_mae': 2.5,             # Energy MAE (kcal/mol)
  'E_rmse': 3.2,            # Energy RMSE (kcal/mol)
  'F_mae': 0.8,             # Forces MAE (kcal/mol/Å)
  'F_rmse': 1.1,            # Forces RMSE (kcal/mol/Å)
  'q_mae': 0.05,            # Charges MAE
  'q_rmse': 0.08,           # Charges RMSE
}
```

### Checkpointing Strategy

**Best Model Checkpointing:**
```python
# Save top 5 models by validation loss
checkpoint:
  save_best: true
  kwargs:
    n_saved: 5
    score_function: -val_loss  # Lower is better
```

**Saved State:**
```python
checkpoint = {
  'model': model.state_dict(),
}
```

Note: Only model weights are saved. Optimizer and scheduler states are not saved to reduce checkpoint size and simplify transfer learning.

**Automatic Saving:**
- After each validation run
- Only if validation loss improves
- Keeps top N checkpoints
- Saved to `checkpoints/{run_name}_best_*.pt`

### Transfer Learning / Continue Training

Resume from checkpoint:

```bash
python -m aimnet2mult.train.cli \
    --config train.yaml \
    --model model.yaml \
    --load checkpoint.pt \   # Restores model weights only
    --save continued.pt \
    ...
```

**What Gets Restored:**
- Model weights (including all fidelity readouts)

Note: Optimizer and scheduler states are not saved in checkpoints. Training will start with a fresh optimizer/scheduler state.

**Fine-Tuning Strategy:**
- Load pretrained weights
- Optionally freeze backbone: `optimizer.force_no_train: ['backbone.*']`
- Train only new fidelity readouts
- Use lower learning rate

### Compilation (Inference-Ready Models)

After training, models are compiled per-fidelity with SAE and optional dispersion:

```
Trained Weights → Extract per-fidelity → Add SAE + Dispersion → TorchScript

   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
   │   Fidelity 0    │     │   Fidelity 1    │     │   Fidelity 2    │
   │   + SAE         │     │   + SAE         │     │   + SAE         │
   │   + D3-BJ       │     │   + D3-BJ       │     │   + None        │
   │   (wB97M)       │     │   (B3LYP)       │     │                 │
   └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
            │                       │                       │
            ▼                       ▼                       ▼
      model_fid0.jpt          model_fid1.jpt          model_fid2.jpt
```

## Self-Atomic Energy (SAE)

### What is SAE?

Self-Atomic Energy (SAE) represents the baseline energy contribution of each atom type. By subtracting SAE during training and adding it back during inference, the neural network only needs to learn the **residual** molecular interaction energy rather than absolute atomic energies.

### How SAE is Calculated

SAE values are computed via **linear regression** on the training dataset:

```
E_molecule = Σ SAE(Z_i) + E_residual
```

The calculation:
1. Build a composition matrix F where F[mol, Z] = count of element Z in molecule
2. Solve the linear system: F × SAE = E (using least squares)
3. Outlier filtering: Remove 2nd and 98th percentile energies, then recalculate

### SAE Calculation Script

```bash
python -m aimnet2mult.train.calc_sae dataset.h5 sae.yaml
```

Output format (`sae.yaml`):
```yaml
1: -13.587234      # Hydrogen
6: -1029.476123    # Carbon
7: -1485.234567    # Nitrogen
8: -2041.789012    # Oxygen
```

### SAE in the Pipeline

```
Training Phase:
  Dataset energies → Subtract SAE → Train on residuals

Inference Phase:
  Model predicts residuals → Add SAE back → Final energy
```

## Dispersion Corrections

### Overview

Dispersion (van der Waals) corrections account for long-range correlation effects not captured by most DFT functionals. The package supports embedding DFT-D3 with Becke-Johnson (BJ) damping directly into compiled models.

### DFT-D3 Theory

The D3-BJ dispersion energy:

```
E_disp = -Σ_AB [ s6 * C6_AB / (R_AB^6 + (a1*√(C8/C6) + a2)^6)
              + s8 * C8_AB / (R_AB^8 + (a1*√(C8/C6) + a2)^8) ]
```

Where:
- **C6, C8**: Dispersion coefficients (depend on coordination numbers)
- **R_AB**: Interatomic distance
- **s6, s8, a1, a2**: Functional-specific damping parameters

### Available Functionals

| Functional | s6   | s8     | a1     | a2     |
|------------|------|--------|--------|--------|
| `b3lyp`    | 1.0  | 1.9889 | 0.3981 | 4.4211 |
| `pbe`      | 1.0  | 0.7875 | 0.4289 | 4.4407 |
| `pbe0`     | 1.0  | 1.2177 | 0.4145 | 4.8593 |
| `wb97x`    | 1.0  | 0.0000 | 0.0000 | 5.4959 |
| `wb97m`    | 1.0  | 0.3908 | 0.5660 | 3.1280 |
| `tpss`     | 1.0  | 1.9435 | 0.4535 | 4.4752 |
| `bp86`     | 1.0  | 3.2822 | 0.3946 | 4.8516 |
| `m062x`    | 1.0  | 0.0000 | 0.0000 | 5.0580 |
| `b973c`    | 1.0  | 1.5000 | 0.3700 | 4.1000 |

### Dispersion Options

When compiling models, specify dispersion via command line:

```bash
# No dispersion
python -m aimnet2mult.tools.compile_jit \
    --dispersion none \
    ...

# D3-BJ with predefined functional
python -m aimnet2mult.tools.compile_jit \
    --dispersion d3bj \
    --dispersion-functional wb97m \
    ...

# D3-BJ with custom parameters
python -m aimnet2mult.tools.compile_jit \
    --dispersion d3bj \
    --dispersion-s6 1.0 \
    --dispersion-s8 0.5 \
    --dispersion-a1 0.4 \
    --dispersion-a2 4.5 \
    ...
```

### Per-Fidelity Dispersion

Different fidelities can have different dispersion settings:

```bash
# Fidelity 0: wB97M-D3 (already includes D3)
--fidelity-level 0 --dispersion d3bj --dispersion-functional wb97m

# Fidelity 1: Pure functional, no dispersion in reference data
--fidelity-level 1 --dispersion none

# Fidelity 2: B3LYP-D3
--fidelity-level 2 --dispersion d3bj --dispersion-functional b3lyp
```

## Long-Range Coulomb Energy

The model includes a long-range Coulomb energy module (`LRCoulomb`) that computes electrostatic interactions from predicted atomic charges.

### Configuration in `model.yaml`

```yaml
lrcoulomb:
  class: aimnet2mult.modules.LRCoulomb
  kwargs:
    key_in: 'charges'
    key_out: 'energy'
    method: 'simple'    # Options: 'simple', 'dsf', 'ewald'
    rc: 4.6             # Cutoff radius (Å) for simple method
```

### Available Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `simple` | Exponential cutoff function | Isolated molecules, fast |
| `dsf` | Damped Shifted Force | MD simulations, force continuity |
| `ewald` | Full Ewald summation | Periodic systems, crystals |

### Method Parameters

**Simple method** (`aimnet2mult/modules.py` lines 277-288):
```python
def coul_simple(self, data):
    fc = 1.0 - ops.exp_cutoff(d_ij, self.rc)  # Smooth cutoff
    e_ij = fc * q_ij / d_ij                    # Coulomb energy
```
- `rc`: Cutoff radius (default: 4.6 Å)

**DSF method** (`aimnet2mult/modules.py` lines 302-312):
```python
def coul_dsf(self, data):
    epot = ops.coulomb_potential_dsf(q_j, d_ij, self.dsf_rc, self.dsf_alpha, data)
    e = e - self.coul_simple_sr(data)  # Subtract short-range to avoid double counting
```
- `dsf_alpha`: Damping parameter (default: 0.2)
- `dsf_rc`: DSF cutoff radius (default: 15.0 Å)

**Ewald method** (`aimnet2mult/modules.py` lines 314-374):
- Requires `cell` data (3x3 unit cell matrix)
- Only works with `nb_mode == 1` (single molecule mode)
- Computes real space + reciprocal space + self-interaction terms

### Example Configurations

```yaml
# Default (simple method)
lrcoulomb:
  class: aimnet2mult.modules.LRCoulomb
  kwargs:
    key_in: 'charges'
    key_out: 'energy'
    method: 'simple'
    rc: 4.6

# DSF for MD simulations
lrcoulomb:
  class: aimnet2mult.modules.LRCoulomb
  kwargs:
    key_in: 'charges'
    key_out: 'energy'
    method: 'dsf'
    dsf_alpha: 0.2
    dsf_rc: 15.0

# Ewald for periodic systems
lrcoulomb:
  class: aimnet2mult.modules.LRCoulomb
  kwargs:
    key_in: 'charges'
    key_out: 'energy'
    method: 'ewald'
```

## Data Format

Molecular data is stored in HDF5 files with molecules grouped by atom count:

```
dataset.h5
├── 3/                          # Molecules with 3 atoms
│   ├── coord                   # (n_mols, 3, 3) float32 - Coordinates (Å)
│   ├── numbers                 # (n_mols, 3) int32 - Atomic numbers
│   ├── charge                  # (n_mols,) float32 - Molecular charge
│   ├── energy                  # (n_mols,) float32 - Energy (eV)
│   ├── forces                  # (n_mols, 3, 3) float32 - Forces [OPTIONAL]
│   ├── charges                 # (n_mols, 3) float32 - Partial charges [OPTIONAL]
│   ├── spin_charges            # (n_mols, 3) float32 - Spin densities [OPTIONAL]
│   └── mult                    # (n_mols,) float32 - Multiplicity [OPTIONAL]
├── 4/
│   └── ...
└── N/
    └── ...
```

## Inference

```python
import torch

# Load compiled model
model = torch.jit.load('model_fid0.jpt', map_location='cpu')

# Prepare input
data = {
    'coord': torch.tensor([[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]]),
    'numbers': torch.tensor([[8, 1, 1]]),
    'charge': torch.tensor([0.0]),
    'mult': torch.tensor([1.0])
}

# Run inference
output = model(data)
print(f"Energy: {output['energy'].item():.6f} eV")
print(f"Forces shape: {output['forces'].shape}")
```

## Configuration

### Training Configuration (`train.yaml`)

Key settings:
```yaml
fidelity_offset: 200              # Atomic number offset between fidelities
use_fidelity_readouts: true       # Fidelity-specific output heads

data:
  fidelity_datasets:
    0: path/to/fidelity_0.h5
    1: path/to/fidelity_1.h5
  fidelity_weights:
    0: 1.0
    1: 1.0
  sae:
    energy:
      files:
        0: path/to/sae_fid0.yaml
        1: path/to/sae_fid1.yaml

wandb:
  init:
    mode: online                  # online, offline, or disabled
    project: my_project
```

## Wandb Integration

Training metrics are automatically logged to wandb:
- Energy/Forces/Charges MAE and RMSE
- Per-fidelity validation metrics
- Learning rate schedule
- System metrics (GPU usage, memory)

View your runs at: https://wandb.ai/your-username/your-project

## Practical Training Tips

### Recommended Training Workflow

**1. Start with Single-Fidelity Training**

Before attempting multi-fidelity:
```bash
# Train on your best dataset first
python -m aimnet2mult.train.cli \
    --config examples/config/train.yaml \
    --model examples/config/model.yaml \
    --save single_fidelity.pt \
    data.fidelity_datasets.0="high_quality.h5" \
    data.sae.energy.files.0="sae_high_quality.yaml" \
    fidelity_offset=200 \
    use_fidelity_readouts=false  # Use base AIMNet2 mode
```

Benefits:
- Faster iteration
- Easier debugging
- Establishes baseline performance
- Can use as pretrained weights for multi-fidelity

**2. Gradually Add Fidelities**

```bash
# Add second fidelity
python -m aimnet2mult.train.cli \
    --config examples/config/train.yaml \
    --model examples/config/model.yaml \
    --load single_fidelity.pt \  # Start from pretrained
    --save two_fidelity.pt \
    data.fidelity_datasets.0="high_quality.h5" \
    data.fidelity_datasets.1="medium_quality.h5" \
    ...
```

**3. Monitor Per-Fidelity Metrics**

Check WandB for each fidelity separately:
- `val/fid0/E_rmse` - Fidelity 0 energy RMSE
- `val/fid1/E_rmse` - Fidelity 1 energy RMSE

If one fidelity performs poorly, adjust `fidelity_weights`.

### Hyperparameter Tuning

**Learning Rate Selection:**

```yaml
# Conservative (stable but slow)
optimizer:
  kwargs:
    lr: 1.0e-5

# Standard (recommended starting point)
optimizer:
  kwargs:
    lr: 5.0e-5

# Aggressive (faster but may diverge)
optimizer:
  kwargs:
    lr: 1.0e-4
```

Use ReduceLROnPlateau to find optimal LR automatically, or use CosineAnnealing with LR range test.

**Batch Size vs Atoms:**

```yaml
# Option 1: Fixed batch size (molecules)
samplers:
  train:
    kwargs:
      batch_mode: molecules
      batch_size: 32

# Option 2: Adaptive by atom count (recommended)
samplers:
  train:
    kwargs:
      batch_mode: atoms
      batch_size: 4096  # Total atoms per batch
```

Atom-based batching balances GPU memory usage across molecular sizes.

**Fidelity Weights:**

```yaml
# Equal weighting (default)
data:
  fidelity_weights:
    0: 1.0
    1: 1.0
    2: 1.0

# Prefer high-fidelity data
data:
  fidelity_weights:
    0: 3.0  # 3x more samples from fidelity 0
    1: 1.0
    2: 0.5  # Half as many from fidelity 2
```

**Loss Component Weights:**

```yaml
loss:
  kwargs:
    components:
      energy:
        weight: 1.0      # Reference
      forces:
        weight: 0.5      # Important but 0.5× energy
      charges:
        weight: 0.01     # Much smaller contribution
      spin_charges:
        weight: 0.01
```

Typical ratios: Energy (1.0), Forces (0.1-1.0), Charges (0.001-0.1).

### Scheduler Recommendations

**For Exploratory Training (unknown optimal schedule):**

```yaml
scheduler:
  class: ignite.handlers.param_scheduler.ReduceLROnPlateauScheduler
  kwargs:
    metric_name: val_loss
    factor: 0.9
    patience: 2
  terminate_on_low_lr: 1.0e-7
```

**For Production Training (known duration):**

```yaml
scheduler:
  class: ignite.handlers.param_scheduler.CosineAnnealingScheduler
  kwargs:
    start_value: 1.0e-4
    end_value: 1.0e-6
    cycle_size: 2000000  # Full training run: 100 epochs × 20k batches
```

**For Finding Optimal LR:**

Use short warm restarts to explore:
```yaml
scheduler:
  class: ignite.handlers.param_scheduler.CosineAnnealingScheduler
  kwargs:
    start_value: 1.0e-4
    end_value: 1.0e-6
    cycle_size: 40000    # 2 epochs
    cycle_mult: 2.0      # Double each cycle
```

See `examples/config/scheduler_examples.yaml` and `examples/SCHEDULER_GUIDE.md` for details.

### Validation Frequency

**Trade-offs:**

```yaml
# Frequent validation (more monitoring, slower training)
log_frequency:
  val: 100   # Every 100 iterations (~0.5% of epoch)

# Standard (good balance)
log_frequency:
  val: 500   # Every 500 iterations (~2.5% of epoch)

# Infrequent (faster training, less monitoring)
log_frequency:
  val: 5000  # Every 5000 iterations (~25% of epoch)
```

**Recommendation:**
- Development: `val: 100-500` (frequent monitoring)
- Production: `val: 1000-5000` (focus on training speed)

### Common Issues

**Issue: Training loss decreases but validation loss increases**

**Solution:** Overfitting. Try:
```yaml
# Increase weight decay
optimizer:
  kwargs:
    weight_decay: 1.0e-5  # Was 1.0e-6

# Reduce model capacity
model:
  num_layers: 3  # Was 4
```

**Issue: Loss is NaN or explodes**

**Solution:** Learning rate too high or gradient overflow. Try:
```yaml
# Reduce learning rate
optimizer:
  kwargs:
    lr: 1.0e-5  # Was 5.0e-5

# Increase gradient clipping (default is 0.4)
# Edit engine.py to increase clip value
```

**Issue: One fidelity performs much worse than others**

**Solution:** Check data quality and adjust weights:
```yaml
# Reduce weight for problematic fidelity
data:
  fidelity_weights:
    0: 1.0
    1: 0.2  # Problematic fidelity gets less weight
    2: 1.0
```

**Issue: Validation metrics don't appear in WandB**

**Solution:** Check configuration:
```yaml
# Ensure val logging is enabled
wandb:
  log_frequency:
    val: 500  # Not null!

# Check metrics are configured
metrics:
  class: aimnet2mult.train.metrics.RegMultiMetric
  kwargs:
    cfg: {...}  # Must be present
```

### Performance Optimization

**GPU Memory Optimization:**

```yaml
# Reduce batch size
data:
  samplers:
    train:
      kwargs:
        batch_size: 2048  # Was 4096

# Use gradient accumulation (future feature)
# Currently: manually reduce batch_size and increase batches_per_epoch
```

**Training Speed:**

```yaml
# Increase workers (if CPU allows)
data:
  loaders:
    train:
      num_workers: 8  # Was 4

# Use pin_memory for GPU
data:
  loaders:
    train:
      pin_memory: true

# Reduce validation frequency
wandb:
  log_frequency:
    val: 2000  # Validate less often
```

**Multi-GPU Training (DDP):**

```bash
# Use torchrun for distributed training
torchrun --nproc_per_node=4 -m aimnet2mult.train.cli \
    --config train.yaml \
    --model model.yaml \
    --save model.pt \
    ...
```

### Debugging Checklist

Before opening an issue, verify:

- [ ] SAE files exist and have correct format
- [ ] HDF5 datasets have required keys (`coord`, `numbers`, `energy`)
- [ ] Atomic numbers in data match SAE file
- [ ] `fidelity_offset` matches compilation settings
- [ ] Validation dataset is separate from training
- [ ] WandB is logged in (`wandb login`)
- [ ] PyTorch can access GPU (`torch.cuda.is_available()`)
- [ ] All required dependencies installed

### Resources

- **Configuration Examples:** `examples/config/`
- **Scheduler Guide:** `examples/SCHEDULER_GUIDE.md`
- **Logging Verification:** `WANDB_LOGGING_VERIFICATION.md`
- **Scheduler Update Summary:** `SCHEDULER_UPDATE_SUMMARY.md`

## License

MIT License - Copyright (c) 2025 AIMNet2Mult Contributors

See [LICENSE](LICENSE) file for full details.

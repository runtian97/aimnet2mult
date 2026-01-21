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

```bash
git clone https://github.com/runtian97/aimnet2mult.git
cd aimnet2mult
pip install -e .

# Optional: Install dispersion packages for D3/D4 support
pip install dftd3-python dftd4

# Optional: Install wandb for experiment tracking
pip install wandb
wandb login
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

## License

MIT License - Copyright (c) 2025 AIMNet2Mult Contributors

See [LICENSE](LICENSE) file for full details.

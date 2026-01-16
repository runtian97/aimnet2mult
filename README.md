# AIMNet2mult - Multi-Fidelity Neural Network Potential

A PyTorch implementation of AIMNet2 with **mixed-fidelity training** support. Train a single model on datasets from different computational methods simultaneously.

## Key Features

- **Mixed-Fidelity Training**: Single model learns from multiple data sources with different accuracy levels
- **Heterogeneous Labels**: Different datasets can have different available properties (energy, forces, charges, etc.)
- **Automatic Masking**: Loss calculation automatically adapts to available labels per sample
- **Transfer Learning**: Load pretrained weights and fine-tune on new datasets
- **TorchScript Export**: Compile models for fast production inference

## Installation

```bash
git clone https://github.com/runtian97/aimnet2mult.git
cd aimnet2mult
pip install .
```

## Package Structure

```
aimnet2mult/
├── aimnet2mult/                    # Core package
│   ├── models/                     # Neural network architectures
│   │   ├── aimnet2.py             # Base AIMNet2 model
│   │   ├── mixed_fidelity_aimnet2.py  # Multi-fidelity wrapper
│   │   ├── base.py                # Abstract base classes
│   │   └── jit_wrapper.py         # TorchScript export utilities
│   │
│   ├── data/                       # Data loading and processing
│   │   ├── base.py                # MultiFidelityDataset base class
│   │   ├── mixed.py               # MixedFidelityDataset (applies atomic number offsets)
│   │   ├── sgdataset.py           # Size-grouped HDF5 dataset
│   │   ├── sampler.py             # Mixed-fidelity batch sampler
│   │   ├── collate.py             # Collate function with automatic masking
│   │   └── loaders.py             # DataLoader creation utilities
│   │
│   ├── train/                      # Training infrastructure
│   │   ├── cli.py                 # Command-line interface
│   │   ├── runner.py              # Training orchestration
│   │   ├── engine.py              # Ignite engines (trainer/evaluator)
│   │   ├── loss.py                # Loss functions with masking support
│   │   ├── metrics.py             # Evaluation metrics
│   │   ├── configuration.py       # Config loading and validation
│   │   ├── calc_sae.py            # Self-atomic energy calculation
│   │   └── utils.py               # Optimizer, scheduler, WandB setup
│   │
│   ├── tools/                      # Utilities
│   │   └── compile_jit.py         # TorchScript compilation
│   │
│   ├── config/                     # Configuration system
│   │   ├── builder.py             # Module instantiation from YAML
│   │   ├── default_model.yaml     # Default model architecture
│   │   └── default_train_mixed_fidelity.yaml  # Default training config
│   │
│   ├── utils/                      # Utilities
│   │   ├── derivatives.py         # Hessian and higher-order derivatives
│   │   ├── units.py               # Unit conversions (Hartree ↔ eV)
│   │   └── imports.py             # Path utilities
│   │
│   ├── modules.py                  # Neural network modules (MLP, Output, etc.)
│   ├── aev.py                      # Atomic environment vectors
│   ├── ops.py                      # Tensor operations
│   ├── nbops.py                    # Neighbor list operations
│   └── constants.py                # Physical constants
│
└── examples/                       # Example scripts and data
    ├── YAML/                       # Configuration files
    │   ├── model.yaml             # Model architecture config
    │   └── train.yaml             # Training parameters config
    ├── fake_dataset/               # Synthetic datasets for testing
    │   ├── fidelity0.h5           # High-fidelity (all labels)
    │   ├── fidelity1.h5           # Mid-fidelity (partial labels)
    │   └── fidelity2.h5           # Low-fidelity (energy + spin only)
    │  
    ├── train.sh                    # Training from scratch
    ├── train_load_weight.sh        # Transfer learning example
    └── train_single_fidelity.sh    # Single-fidelity training
```

## Quick Start

### Training

See example scripts in `examples/`:

```bash
# Train from scratch with 3 fidelities
bash examples/train.sh

# Transfer learning from pretrained model
bash examples/train_load_weight.sh

# Single-fidelity training
bash examples/train_single_fidelity.sh
```

**Configuration files:**
- `examples/YAML/train.yaml` - Training parameters, dataset paths, loss weights
- `examples/YAML/model.yaml` - Model architecture (layer sizes, features, etc.)

## Data Format

Molecular data is stored in HDF5 files with molecules grouped by atom count:

```
dataset.h5
├── 3/                          # Molecules with 3 atoms
│   ├── coord                   # (n_mols, 3, 3) float32 - Cartesian coordinates (Å)
│   ├── numbers                 # (n_mols, 3) int32 - Atomic numbers
│   ├── charge                  # (n_mols,) float32 - Total molecular charge
│   ├── energy                  # (n_mols,) float32 - Total energy (eV) 
│   ├── forces                  # (n_mols, 3, 3) float32 - Atomic forces (eV/Å) [OPTIONAL]
│   ├── charges                 # (n_mols, 3) float32 - Partial atomic charges (e) [OPTIONAL]
│   ├── spin_charges            # (n_mols, 3) float32 - Spin densities (e) [OPTIONAL]
│   └── mult                    # (n_mols,) float32 - Spin multiplicity [OPTIONAL, defaults to 1]
│
├── 4/                          # Molecules with 4 atoms
│   ├── coord                   # (n_mols, 4, 3) float32
│   ├── numbers                 # (n_mols, 4) int32
│   └── ...
│
└── N/                          # Molecules with N atoms
    └── ...
```

**Notes:**
- `coord`, `numbers`, `charge`, and `energy` are required fields
- All other labels (`forces`, `charges`, `spin_charges`, `mult`) are optional
- Different datasets can have different combinations of optional labels
- Missing labels are automatically masked during training

### Inference

#### Load TorchScript Model

```python
import torch
from openbabel import pybel

# Load compiled model
filename = '/Users/nickgao/Desktop/pythonProject/aimnet2mult/examples/run/model_fid1.jpt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(filename, map_location=device)

# Load molecule
mol_file = '/Users/nickgao/Desktop/pythonProject/local_code_template/test_sdfs/CAMVES_a.sdf'
mol = next(pybel.readfile('sdf', mol_file))

# Prepare input
coord = torch.as_tensor([a.coords for a in mol.atoms]).unsqueeze(0).to(device)
numbers = torch.as_tensor([a.atomicnum for a in mol.atoms]).unsqueeze(0).to(device)
charge = torch.as_tensor([mol.charge]).to(device)
mult = torch.as_tensor([mol.spin]).to(device)

_in = dict(
    coord=coord,
    numbers=numbers,
    charge=charge,
    mult=mult,
)

# Run inference
_out = model(_in)
energy = _out['energy']         # Total energy (eV)
forces = _out['forces']         # Atomic forces (eV/Å)
charges = _out['charges']       # Partial charges (e)
spin = _out['spin_charges']     # Spin densities (e)
```

#### Compute Hessian (Second Derivatives)

```python
import torch
from openbabel import pybel
from aimnet2mult.models.jit_wrapper import load_jit_model

# Load the TorchScript model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "/Users/nickgao/Desktop/pythonProject/aimnet2mult/examples/run/model_fid1.jpt"

model = load_jit_model(model_path)
mol_path = '/Users/nickgao/Desktop/pythonProject/local_code_template/test_sdfs/CAMVES_a.sdf'

# Load molecule
mol = next(pybel.readfile('sdf', mol_path))

# Prepare input with gradient tracking
coord = torch.as_tensor([a.coords for a in mol.atoms]).unsqueeze(0).to(device)
coord.requires_grad_(True)  # Enable gradient for Hessian computation
numbers = torch.as_tensor([a.atomicnum for a in mol.atoms]).unsqueeze(0).to(device)
charge = torch.as_tensor([mol.charge]).to(device)
mult = torch.as_tensor([mol.spin]).to(device)

_in = {"coord": coord, "numbers": numbers, "charge": charge, "mult": mult}

# Predict forces and Hessian
_out = model(_in, compute_hessian=True, hessian_mode="autograd")

print("Forces shape:", _out["forces"])
print("Hessian:", _out["hessian"])
print(_out.keys())
```

## Core Concepts

### Mixed-Fidelity Training

Train on datasets from different computational methods simultaneously:

- **Atomic Number Offsetting**: Element Z becomes Z, Z+200, Z+400 for fidelities 0, 1, 2 (configurable via `fidelity_offset` in training config)
- **Fidelity-Specific Layers**: Each fidelity has its own readout layers and embeddings
- **Weighted Sampling**: Control how often each fidelity is sampled during training

#### How Atomic Number Shifting Works

The atomic number shifting mechanism differs across training, compilation, and inference to enable multi-fidelity learning while maintaining a clean user interface.

**Example: Water molecule (O, H, H) in Fidelity 1 with offset=200**

##### Training
- **Data loading** shifts both atomic numbers and SAE keys by the fidelity offset
- Model receives pre-shifted inputs and uses them directly

```
Input atomic numbers: [8, 1, 1] (original)
  ↓ Shift +200 (data layer)
Training data: [208, 201, 201] (shifted)
  ↓
Model embedding lookup: afv[208], afv[201], afv[201] ✓
SAE keys: {201: -0.5, 208: -75.0} (also shifted +200)
  ↓
Direct SAE lookup for shifted numbers ✓
```

##### Compilation
- **Extracts** fidelity-specific weights and **restores** AFV embeddings to standard indexing
- **Shifts** SAE keys during storage

```
Trained model AFV [size 728]:
  Rows 200-328 (fidelity 1 embeddings)
    ↓ Extract and remap
Compiled model AFV [size 128]:
  Rows 0-128 (standard indexing)
  afv[201] → afv[1], afv[208] → afv[8] ✓

Original SAE: {1: -0.5, 8: -75.0}
    ↓ Shift +200 during storage
Compiled SAE storage: {201: -0.5, 208: -75.0} ✓
```

##### Inference
- User provides **standard** atomic numbers
- Base model uses standard indexing
- SAE wrapper shifts numbers only for SAE lookup

```
User input: [8, 1, 1] (standard)
  ↓
Base model: afv[8], afv[1], afv[1] (standard lookup) ✓
  ↓
SAE wrapper: [8, 1, 1] + 200 = [208, 201, 201]
  ↓
SAE lookup: sae_tensor[208], sae_tensor[201] ✓
  ↓
Final energy = model_output + SAE_correction
```

**Key Insight**: The compiled model provides a standard interface (normal atomic numbers in/out) while internally using shifted SAE storage for consistency with the training procedure.

### Heterogeneous Labels

Different datasets can have different available properties:

| Dataset | energy | forces | charges | spin | mult |
|---------|--------|--------|---------|------|------|
| Fid0    | ✓      | ✓      | ✓       | ✓    | ✓    |
| Fid1    | ✓      | ✓      | ✓       | ✗    | ✗    |
| Fid2    | ✓      | ✗      | ✗       | ✓    | ✗    |

The collate function automatically creates masks, and loss functions skip missing labels.

### Self-Atomic Energy (SAE)

Removes atomic baseline energies before training:

```bash
# Calculate SAE for each dataset
python -m aimnet2mult.train.calc_sae dataset.h5 sae.yaml
```

SAE is applied during data loading: `E_corrected = E_raw - Σ SAE[atom_i]`


## License

MIT License - Copyright (c) 2025 AIMNet2Mult Contributors

See [LICENSE](LICENSE) file for full details.


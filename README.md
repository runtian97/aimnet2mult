# AIMNet2Mult

AIMNet2 Multi-Fidelity Training Package for molecular property prediction using neural networks.

## Overview

AIMNet2Mult is a PyTorch-based package for training and using neural network models to predict molecular properties at multiple levels of theoretical fidelity. The package supports prediction of energies, forces, atomic charges, and Hessians for molecular systems.

## Installation

### Quick Install from GitHub

```bash
pip install git+https://github.com/runtian97/aimnet2mult.git
```

### Install from Source

**Option 1: Editable/Development Mode** (recommended for development)

```bash
git clone https://github.com/runtian97/aimnet2mult.git
cd aimnet2mult
pip install -e .
```

**Option 2: Regular Installation**

```bash
git clone https://github.com/runtian97/aimnet2mult.git
cd aimnet2mult
pip install .
```

**Option 3: Install with Training Dependencies**

```bash
pip install -e ".[train]"  # or pip install ".[train]" for non-editable
```

**Option 4: Install with All Optional Dependencies**

```bash
pip install -e ".[all]"  # or pip install ".[all]" for non-editable
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.1.0
- NumPy >= 1.20.0
- h5py >= 3.0.0
- PyYAML >= 5.0
- OmegaConf >= 2.0
- opt_einsum >= 3.0

For training:
- pytorch-ignite >= 0.5.0
- wandb >= 0.12.0 (optional, for experiment tracking)

## Quick Start - Using Trained Models

### Basic Example

```python
import torch
from openbabel import pybel

# Load the JIT-compiled model
filename = 'model.jpt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(filename, map_location=device)

# Read molecule from file (SDF, MOL2, PDB, etc.)
mol_file = 'molecule.sdf'
mol = next(pybel.readfile('sdf', mol_file))

# Extract molecular information
coord = torch.as_tensor([a.coords for a in mol.atoms]).unsqueeze(0).to(device)
numbers = torch.as_tensor([a.atomicnum for a in mol.atoms]).unsqueeze(0).to(device)
charge = torch.as_tensor([mol.charge]).to(device)
mult = torch.as_tensor([mol.spin]).to(device)

# Prepare input dictionary
_in = dict(
    coord=coord,
    numbers=numbers,
    charge=charge,
    mult=mult,
)

# Run prediction
_out = model(_in)

# Access predictions
print("Energy:", _out['energy'])
print("Atomic charges:", _out['charges'])
print("Spin charges:", _out['spin_charges'])
```

### Computing Forces and Hessian

```python
import torch
from openbabel import pybel
from aimnet2mult.models.jit_wrapper import load_jit_model

# Load model with wrapper (enables force/Hessian computation)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_jit_model('model.jpt', device=device)

# Read molecule
mol = next(pybel.readfile('sdf', 'molecule.sdf'))

# Convert to tensors
coord = torch.as_tensor([a.coords for a in mol.atoms]).unsqueeze(0).to(device)
numbers = torch.as_tensor([a.atomicnum for a in mol.atoms]).unsqueeze(0).to(device)
charge = torch.as_tensor([mol.charge]).to(device)
mult = torch.as_tensor([mol.spin]).to(device)

_in = {
    "coord": coord,
    "numbers": numbers,
    "charge": charge,
    "mult": mult
}

# Compute forces and Hessian (wrapper handles gradient computation automatically)
_out = model(_in, compute_hessian=True, hessian_mode="autograd")

print("Energy:", _out['energy'])
print("Forces shape:", _out['forces'].shape)  # (batch_size, n_atoms, 3)
print("Hessian shape:", _out['hessian'].shape)  # (batch_size, n_atoms, 3, n_atoms, 3)
```

### Understanding Model Outputs

```python
# After running prediction with _out = model(_in)

# Molecular properties
energy = _out['energy']              # Total molecular energy
charges = _out['charges']            # Atomic charges
spin_charges = _out['spin_charges']  # Atomic spin densities

# Molecular geometry
coord = _out['coord']                # Atomic coordinates (input echoed)
numbers = _out['numbers']            # Atomic numbers (input echoed)
natom = _out['_natom']              # Number of atoms

# Atomic features
a = _out['a']                        # Atomic feature vectors
aim = _out['aim']                    # AIMNet2 convolution outputs (local environment)

# Geometric descriptors
d_ij = _out['d_ij']                  # Pairwise distance matrix
gs = _out['gs']                      # Scalar geometric features
gv = _out['gv']                      # Vector geometric features

# Derived features
afv = _out['a'].flatten(-2, -1)     # Atomic feature vector (flattened)
```

### Reading Molecular Information from Files

The package works with various molecular file formats through OpenBabel:

```python
from openbabel import pybel

# Supported formats: SDF, MOL2, PDB, XYZ, etc.
mol = next(pybel.readfile('sdf', 'molecule.sdf'))

# Extract molecular properties
coordinates = [a.coords for a in mol.atoms]      # List of [x, y, z] for each atom
atomic_numbers = [a.atomicnum for a in mol.atoms] # Atomic numbers (6=C, 1=H, etc.)
molecular_charge = mol.charge                     # Total charge
spin_multiplicity = mol.spin                      # 1=singlet, 2=doublet, 3=triplet, etc.

# Convert to PyTorch tensors
import torch
coord = torch.as_tensor(coordinates).unsqueeze(0)
numbers = torch.as_tensor(atomic_numbers).unsqueeze(0)
charge = torch.as_tensor([molecular_charge])
mult = torch.as_tensor([spin_multiplicity])
```

## Model Features

The trained models provide rich molecular representations:

- **energy**: Total molecular energy
- **forces**: Atomic forces (gradients of energy w.r.t. coordinates)
- **hessian**: Second derivatives (for vibrational analysis)
- **charges**: Atomic charges
- **spin_charges**: Atomic spin densities
- **a**: Atomic feature vectors
- **aim**: AIMNet2 convolution outputs (local environment encoding)
- **d_ij**: Interatomic distances
- **gs**: Scalar geometric features
- **gv**: Vector geometric features

### Optional D3 Dispersion Correction

The package includes implementations of DFTD3-BJ dispersion correction and D3-TS (Tkatchenko-Scheffler) combination rules that can be optionally configured during model training for improved accuracy of long-range interactions. These corrections are available as modular components that can be added to the model architecture through the configuration files.

## Training Your Own Models

Training examples including example datasets and input configuration files are available in the `examples/` folder.

### Configuration Files

See `examples/mismatched/` for example configuration:
- `train.yaml`: Training configuration (data paths, hyperparameters, learning rates, etc.)
- `model.yaml`: Model architecture configuration (layers, features, dispersion corrections, etc.)

### Training Data Format

Training data should be in HDF5 format containing:
- Molecular geometries (coordinates and atomic numbers)
- Target properties (energies, forces, charges, etc.)
- Metadata (charge, spin multiplicity)

### Run Training

```bash
cd examples
bash train.sh
```

The training script will:
1. Load configuration from YAML files
2. Initialize the model architecture
3. Load training and validation datasets
4. Train with the specified hyperparameters
5. Save checkpoints and final model

### Custom Training

```python
from aimnet2mult.train import train_mixed_fidelity

# Configure and run training
# See examples/mismatched/train.yaml for all configuration options
```

## Examples

The `examples/` directory contains:

- **`aim_aev_afv.ipynb`**: Jupyter notebook demonstrating model loading, inference, and accessing all model outputs
- **`train.sh`**: Shell script for training models
- **`mismatched/`**: Example training configuration files
  - `train.yaml`: Complete training setup
  - `model.yaml`: Model architecture definition

See `get_hess_force.py` in the root directory for a complete example of computing forces and Hessians.

## Package Structure

```
aimnet2mult/
├── models/              # Model architectures and JIT wrapper
├── data/                # Dataset and data loading utilities
├── train/               # Training loops and utilities
├── tools/               # Helper tools (JIT compilation, etc.)
├── config/              # Configuration builders
├── utils/               # General utilities
├── modules.py           # Neural network modules (including D3 correction)
├── aev.py              # Atomic environment vector computation
├── ops.py              # Core tensor operations
├── nbops.py            # Neighbor list operations
├── constants.py        # Physical constants and parameters
└── d3bj_data.pt        # D3 dispersion correction parameters

examples/
├── mismatched/         # Example training configuration
│   ├── train.yaml     # Training parameters
│   └── model.yaml     # Model architecture
├── train.sh           # Training script
└── aim_aev_afv.ipynb  # Jupyter notebook demonstrating model usage

sample_data/           # Example datasets
get_hess_force.py      # Example: computing forces and Hessians
```

## Citation

If you use AIMNet2Mult in your research, please cite the relevant publications.

## License

MIT License - see LICENSE file for details.

## Support

For questions and issues, please open an issue on the GitHub repository.

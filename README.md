# AIMNet2Mult

AIMNet2 Multi-Fidelity Training Package for molecular modeling and quantum chemistry predictions.

## Installation

```bash
pip install git+https://github.com/runtian97/aimnet2mult.git
```

### From Source

```bash
git clone https://github.com/runtian97/aimnet2mult.git
cd aimnet2mult
pip install -e .
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.1.0
- See `requirements.txt` for full dependencies

## Quick Start

```python
import torch
from aimnet2mult.models.jit_wrapper import load_jit_model

# Load model
model = load_jit_model('path/to/model.jpt')

# Prepare input
coords = torch.randn(1, 10, 3)  # Molecular coordinates
numbers = torch.tensor([[6, 1, 1, 1, 1, 0, 0, 0, 0, 0]])  # Atomic numbers
charge = torch.tensor([0.0])

# Run inference
with torch.no_grad():
    results = model(coords, numbers, charge)
    energy = results['energy']
    forces = results['forces']
```

## Features

- Multi-fidelity learning for molecular modeling
- Atomic environment vectors (AEV)
- D3 dispersion corrections
- Size-grouped datasets for efficient training
- JIT compilation support

## License

MIT License

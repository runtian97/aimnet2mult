# Multi-Fidelity AIMNet2 Training Example

This example demonstrates end-to-end training of a multi-fidelity AIMNet2 model with three datasets that have different available labels.

## Overview

The example uses three fidelity levels:

| Fidelity | Dataset | Available Labels |
|----------|---------|------------------|
| 0 | aimnet2_nse | energy, forces, charges, spin_charges |
| 1 | ani1ccx | energy only |
| 2 | omol_25 | energy, forces, charges, spin_charges |

The key feature demonstrated is **per-molecule masking**: the loss function automatically handles missing labels on a per-molecule basis within mixed batches.

## Quick Start

```bash
# Run the full example
./run_example.sh
```

See `run_example.sh` for the full step-by-step workflow.

## Directory Structure

```
end_to_end_example/
├── config/
│   ├── model.yaml          # Model architecture
│   └── train.yaml          # Training configuration
├── output/
│   ├── subsets/            # Subset datasets (~5K samples each)
│   │   ├── fid0_subset.h5
│   │   ├── fid1_subset.h5
│   │   ├── fid2_subset.h5
│   │   └── sae_fid*.yaml   # SAE corrections
│   ├── checkpoints/        # Training checkpoints (generated)
│   └── compiled/           # Compiled models (generated)
├── run_example.sh          # Main script
└── README.md
```

## Data Requirements

### Required Inputs (x)
- `coord` - Atomic coordinates (Å)
- `numbers` - Atomic numbers
- `charge` - Molecular charge

### Optional Inputs
- `mult` - Spin multiplicity (defaults to 1 if missing)

### Required Labels (y)
- `energy` - Molecular energy (eV)

### Optional Labels (masked if missing)
- `forces` - Atomic forces (eV/Å)
- `charges` - Atomic partial charges (e)
- `spin_charges` - Atomic spin charges (e)

## Per-Molecule Masking

The masking mechanism works as follows:

1. **Collate**: Creates `{key}_mask` tensors indicating which molecules have each label
2. **Loss**: Computes loss per-molecule, then averages only over valid molecules
3. **Gradients**: Flow only through molecules with valid labels

Example with a mixed batch:
```
Batch: [mol0_fid0, mol1_fid1, mol2_fid2]

forces_mask = [1, 0, 1]  # mol1 (ani1ccx) has no forces

forces_loss = (loss_mol0 * 1 + loss_mol1 * 0 + loss_mol2 * 1) / 2
```

## Configuration

### Training Config (`config/train.yaml`)

Key settings:
```yaml
# Data keys with masking
data:
  x: [coord, numbers, charge, mult]
  y: [energy, forces, charges, spin_charges]  # Optional labels masked if missing

# Loss with per-molecule masking
loss:
  class: aimnet2mult.train.loss.MTLoss
  kwargs:
    components:
      energy:
        fn: aimnet2mult.train.loss.energy_loss_fn
        weight: 1.0
      forces:
        fn: aimnet2mult.train.loss.peratom_loss_fn
        weight: 0.2
        kwargs:
          key_true: forces
          key_pred: forces
      # ... charges, spin_charges
```

### Model Config (`config/model.yaml`)

The model includes output heads for all predictions:
- `energy_mlp` - Energy prediction
- `charges` - Partial charges
- `spin_charges` - Spin charges
- `atomic_shift` - Per-element energy shifts
- `atomic_sum` - Sum atomic to molecular energy

## Dispersion Options

Dispersion correction is added at compile time (not during training).

### Dispersion Types (`--dispersion`)
| Type | Description |
|------|-------------|
| `none` | No dispersion correction |
| `d3bj` | DFT-D3 with Becke-Johnson damping |
| `d4` | DFT-D4 (requires dftd4 package) |

### Built-in Functionals (`--dispersion-functional`)
| Functional | s6 | s8 | a1 | a2 |
|------------|-----|------|-------|-------|
| `wb97m` | 1.0 | 0.3908 | 0.566 | 3.128 |
| `wb97x` | 1.0 | 0.0 | 0.464 | 1.0 |
| `b3lyp` | 1.0 | 1.9889 | 0.3981 | 4.4211 |
| `pbe` | 1.0 | 0.7875 | 0.4289 | 4.4407 |
| `pbe0` | 1.0 | 1.2177 | 0.4145 | 4.8593 |

Additional functionals available with the `dftd3` Python package.

## Using Compiled Models

```python
import torch

# Load compiled model
model = torch.jit.load('output/compiled/model_fid0.jpt')

# Prepare input
data = {
    'coord': torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),  # (batch, atoms, 3)
    'numbers': torch.tensor([[6, 1]]),  # (batch, atoms)
    'charge': torch.tensor([0.0]),  # (batch,)
}

# Run inference
with torch.no_grad():
    result = model(data)

print(f"Energy: {result['energy'].item():.4f} eV")
print(f"Charges: {result['charges']}")
```

## Troubleshooting

### Memory Issues
- Reduce `batch_size` in config
- Reduce `batches_per_epoch` for faster iteration
- Use subset datasets (included in this example)

### Missing Labels Error
If you see `KeyError: 'forces'`, ensure:
1. The loss function uses `peratom_loss_fn` which handles missing keys
2. The `y` list in config includes the label

### NaN in Validation
- Small validation sets can cause unstable metrics
- Increase `val_fraction` or use separate validation datasets

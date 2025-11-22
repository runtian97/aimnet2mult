#!/usr/bin/env python
"""
Get Hessian and Forces using optimized AIMNet2Mult JIT wrapper.

This script demonstrates the OPTIMIZED Hessian calculation using
component-by-component differentiation (from aimnet2calc logic).
"""

import torch
from openbabel import pybel
from aimnet2mult.models.jit_wrapper import load_jit_model

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "/Users/nickgao/Desktop/pythonProject/aimnet2mult/examples/run/model_fid1.jpt"
mol_path = '/Users/nickgao/Desktop/pythonProject/local_code_template/test_sdfs/CAMVES_a.sdf'


# Load JIT model (optimized with autograd Hessian by default)
model = load_jit_model(model_path, device=device)

# Load molecule
mol = next(pybel.readfile('sdf', mol_path))


# Prepare input data (batch dimension required)
coord = torch.as_tensor([a.coords for a in mol.atoms], dtype=torch.float32).unsqueeze(0).to(device)
numbers = torch.as_tensor([a.atomicnum for a in mol.atoms], dtype=torch.int64).unsqueeze(0).to(device)
charge = torch.as_tensor([mol.charge], dtype=torch.float32).to(device)
mult = torch.as_tensor([mol.spin], dtype=torch.float32).to(device)

data = {
    "coord": coord,
    "numbers": numbers,
    "charge": charge,
    "mult": mult
}


# Calculate forces and Hessian with OPTIMIZED autograd method
output = model(data, compute_hessian=True, hessian_mode="autograd")

# Optional: Remove batch dimension for single molecule
energy_single = output['energy'].item()
forces_single = output['forces'].squeeze(0)  # (natoms, 3)
hessian_single = output['hessian'].squeeze(0)  # (natoms, 3, natoms, 3)

print(hessian_single)

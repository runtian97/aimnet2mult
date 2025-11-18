#!/usr/bin/env python
"""Create test datasets for multifidelity training."""

import h5py
import numpy as np
import os

def create_test_dataset(filename, n_molecules=100, max_atoms=20):
    """Create a test dataset in SizeGroupedDataset format."""

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with h5py.File(filename, 'w') as f:
        # Create groups for different molecule sizes
        # Group names MUST be integers for SizeGroupedDataset
        for size in range(5, max_atoms + 1, 5):  # 5, 10, 15, 20 atoms
            group = f.create_group(str(size))  # Use integer as string
            n_mols = n_molecules // 4

            # coord: [n_molecules, n_atoms, 3]
            coords = np.random.randn(n_mols, size, 3).astype(np.float32) * 3.0
            group.create_dataset('coord', data=coords)

            # numbers: [n_molecules, n_atoms] - atomic numbers (1-8: H, C, N, O, F, ...)
            numbers = np.random.randint(1, 9, size=(n_mols, size), dtype=np.int32)
            group.create_dataset('numbers', data=numbers)

            # charge: [n_molecules]
            charges = np.zeros(n_mols, dtype=np.float32)
            group.create_dataset('charge', data=charges)

            # mult: [n_molecules] - spin multiplicity (1=singlet, 2=doublet, 3=triplet)
            # Create varied multiplicities
            mult = np.random.choice([1, 2, 3], size=n_mols, p=[0.5, 0.3, 0.2]).astype(np.int32)
            group.create_dataset('mult', data=mult)

            # energy: [n_molecules]
            energies = np.random.randn(n_mols).astype(np.float32) * 10.0 - 500.0
            group.create_dataset('energy', data=energies)

            # forces: [n_molecules, n_atoms, 3]
            forces = np.random.randn(n_mols, size, 3).astype(np.float32) * 0.1
            group.create_dataset('forces', data=forces)

            # charges: [n_molecules, n_atoms] - atomic partial charges
            atom_charges = np.random.randn(n_mols, size).astype(np.float32) * 0.3
            # Ensure they sum to molecular charge
            atom_charges -= atom_charges.mean(axis=1, keepdims=True)
            group.create_dataset('charges', data=atom_charges)

            # spin_charges: [n_molecules, n_atoms] - correlated with multiplicity
            # For singlets (mult=1), spin_charges should be ~0
            # For doublets/triplets, spin_charges should be non-zero
            spin_charges = np.zeros((n_mols, size), dtype=np.float32)
            for i in range(n_mols):
                if mult[i] > 1:
                    # Non-zero spin charges for open-shell systems
                    spin_charges[i] = np.random.randn(size).astype(np.float32) * 0.3
                    # Ensure they roughly sum to (mult-1)
                    spin_charges[i] = spin_charges[i] - spin_charges[i].mean() + (mult[i] - 1) / size
                else:
                    # Near-zero spin charges for closed-shell singlets
                    spin_charges[i] = np.random.randn(size).astype(np.float32) * 0.01
            group.create_dataset('spin_charges', data=spin_charges)

    print(f"Created {filename}")

def main():
    # Create data directory
    data_dir = 'sample_data/mixed'
    os.makedirs(data_dir, exist_ok=True)

    # Create 3 fidelity datasets with different "noise" levels
    # Fidelity 0: highest accuracy (less noise)
    np.random.seed(42)
    create_test_dataset(f'{data_dir}/fidelity0.h5', n_molecules=200, max_atoms=20)

    # Fidelity 1: mid-level
    np.random.seed(43)
    create_test_dataset(f'{data_dir}/fidelity1.h5', n_molecules=400, max_atoms=20)

    # Fidelity 2: lower accuracy (more noise, but more data)
    np.random.seed(44)
    create_test_dataset(f'{data_dir}/fidelity2.h5', n_molecules=800, max_atoms=20)

    print("\nDataset statistics:")
    for i in range(3):
        filename = f'{data_dir}/fidelity{i}.h5'
        with h5py.File(filename, 'r') as f:
            n_groups = len(f.keys())
            total_mols = sum(f[g]['energy'].shape[0] for g in f.keys())
            print(f"  Fidelity {i}: {total_mols} molecules in {n_groups} size groups")

    print("\nTest datasets created successfully!")

if __name__ == '__main__':
    main()

"""Create fake multi-fidelity datasets for testing."""

import os
import h5py
import yaml
import numpy as np
import argparse


def create_molecule(num_atoms, seed=None):
    if seed is not None:
        np.random.seed(seed)
    elements = [1, 6, 7, 8, 9]
    numbers = np.random.choice(elements, size=num_atoms).astype(np.int64)
    coord = np.random.randn(num_atoms, 3).astype(np.float32) * 2.0
    charge = np.random.choice([-1, 0, 1]).astype(np.float32)
    mult = np.random.choice([1, 2, 3]).astype(np.float32)
    return numbers, coord, charge, mult


def compute_fake_properties(numbers, coord, charge, mult, noise_scale=0.1):
    num_atoms = len(numbers)
    atomic_energies = {1: -0.5, 6: -37.8, 7: -54.5, 8: -75.0, 9: -99.7}
    energy = sum(atomic_energies.get(z, -50.0) for z in numbers)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            r = np.linalg.norm(coord[i] - coord[j])
            if r > 0.5:
                energy += -1.0 / r
    energy += np.random.randn() * noise_scale
    forces = np.random.randn(num_atoms, 3).astype(np.float32) * noise_scale
    charges = np.random.randn(num_atoms).astype(np.float32) * 0.2
    spin_charges = np.random.randn(num_atoms).astype(np.float32) * 0.1 * ((mult - 1) / 2)
    return np.float32(energy), forces, charges, spin_charges


def create_dataset(output_path, num_molecules, atoms_range=(3, 15), noise_scale=0.1, seed=42):
    np.random.seed(seed)
    groups = {}
    for mol_idx in range(num_molecules):
        num_atoms = np.random.randint(atoms_range[0], atoms_range[1] + 1)
        numbers, coord, charge, mult = create_molecule(num_atoms, seed=seed + mol_idx)
        energy, forces, charges, spin_charges = compute_fake_properties(
            numbers, coord, charge, mult, noise_scale
        )
        if num_atoms not in groups:
            groups[num_atoms] = {
                'coord': [], 'numbers': [], 'charge': [], 'mult': [],
                'energy': [], 'forces': [], 'charges': [], 'spin_charges': []
            }
        groups[num_atoms]['coord'].append(coord)
        groups[num_atoms]['numbers'].append(numbers)
        groups[num_atoms]['charge'].append(charge)
        groups[num_atoms]['mult'].append(mult)
        groups[num_atoms]['energy'].append(energy)
        groups[num_atoms]['forces'].append(forces)
        groups[num_atoms]['charges'].append(charges)
        groups[num_atoms]['spin_charges'].append(spin_charges)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        for num_atoms, data in groups.items():
            grp = f.create_group(str(num_atoms))
            for key, values in data.items():
                grp.create_dataset(key, data=np.array(values))


def create_sae_file(output_path, fidelity):
    base_sae = {1: -0.500, 6: -37.800, 7: -54.500, 8: -75.000, 9: -99.700}
    sae = {z: e + fidelity * 0.1 for z, e in base_sae.items()}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(sae, f)


def main():
    parser = argparse.ArgumentParser(description='Create fake multi-fidelity datasets')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--num-fidelities', type=int, default=2, help='Number of fidelities')
    parser.add_argument('--num-molecules', type=int, default=500, help='Molecules per fidelity')
    args = parser.parse_args()

    for fid in range(args.num_fidelities):
        dataset_path = os.path.join(args.output, f'fidelity_{fid}.h5')
        sae_path = os.path.join(args.output, f'sae_fid{fid}.yaml')

        num_mols = args.num_molecules if fid == 0 else args.num_molecules * 2
        noise = 0.05 if fid == 0 else 0.2
        seed = 42 + fid * 100

        create_dataset(dataset_path, num_mols, (3, 12), noise, seed)
        create_sae_file(sae_path, fid)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Compute Self-Atomic Energies (SAE) for AIMNet2 datasets."""

import argparse
import os
from collections import defaultdict

import h5py
import numpy as np
import yaml

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None  # Optional dependency when --config is unused


def compute_sae(dataset_path, output_path):
    """
    Compute SAE using a per-atom average of molecular energies.

    Assumes input energies are already in eV (matching AIMNet2 convention).
    """
    atomic_energies = defaultdict(list)

    with h5py.File(dataset_path, 'r') as handle:
        for group_name in handle.keys():
            group = handle[group_name]
            energies = group['energy'][:]
            numbers = group['numbers'][:]

            for energy, nums in zip(energies, numbers):
                # Energy is already in eV
                natoms = np.count_nonzero(nums)
                if natoms == 0:
                    continue
                per_atom = energy / natoms
                for z in nums:
                    if z > 0:  # Ignore padding
                        atomic_energies[int(z)].append(per_atom)

    sae_dict = {z: float(np.mean(values)) for z, values in atomic_energies.items()}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as out_handle:
        yaml.safe_dump(sae_dict, out_handle)

    print(f"Computed SAE for {len(sae_dict)} elements from {dataset_path}")
    print(f"Saved to: {output_path}")
    return sae_dict


def compute_from_config(config_path, output_dir=None):
    """Compute SAE files for every fidelity listed in a training config."""
    if OmegaConf is None:
        raise RuntimeError("omegaconf is required to use --config")

    cfg = OmegaConf.load(config_path)
    fidelity_cfg = cfg.data.get('fidelity_datasets', {})
    fidelity_datasets = {int(fid_key): dataset for fid_key, dataset in fidelity_cfg.items()}

    if not fidelity_datasets:
        raise ValueError(f"No fidelity datasets defined in {config_path}")

    # Get config file directory to resolve relative paths
    config_dir = os.path.dirname(os.path.abspath(config_path))

    sae_files = {}
    for fid, dataset in fidelity_datasets.items():
        # Resolve dataset path relative to config file
        if not os.path.isabs(dataset):
            dataset_path = os.path.abspath(os.path.join(config_dir, dataset))
        else:
            dataset_path = os.path.abspath(dataset)

        target_dir = os.path.abspath(output_dir) if output_dir is not None else os.path.dirname(dataset_path)
        output_path = os.path.join(target_dir, f"sae_fid{fid}.yaml")
        sae_files[fid] = compute_sae(dataset_path, output_path)

    if 0 in sae_files:
        # Resolve fidelity 0 path relative to config file
        fid0_path = fidelity_datasets[0]
        if not os.path.isabs(fid0_path):
            fid0_dataset = os.path.abspath(os.path.join(config_dir, fid0_path))
        else:
            fid0_dataset = os.path.abspath(fid0_path)

        target_dir = os.path.abspath(output_dir) if output_dir is not None else os.path.dirname(fid0_dataset)
        fid0_output = os.path.join(target_dir, "sae.yaml")
        with open(fid0_output, 'w') as out_handle:
            yaml.safe_dump(sae_files[0], out_handle)
        print(f"Duplicated fidelity 0 SAE to {fid0_output} for backwards compatibility.")

    return sae_files


def parse_args():
    parser = argparse.ArgumentParser(description="Compute self-atomic energy (SAE) corrections.")
    parser.add_argument('--dataset', type=str, help='Path to a single fidelity dataset (HDF5).')
    parser.add_argument('--output', type=str, help='Output path for the SAE YAML (used with --dataset).')
    parser.add_argument('--config', type=str, help='Training config containing data.fidelity_datasets.')
    parser.add_argument('--output-dir', type=str, help='Directory to write SAE YAML files when using --config.')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        compute_from_config(args.config, args.output_dir)
        return

    if args.dataset:
        output_path = args.output or os.path.join(os.path.dirname(os.path.abspath(args.dataset)), "sae.yaml")
        compute_sae(args.dataset, output_path)
        return

    dataset_path = 'sample_data/mixed/fidelity0.h5'
    output_path = 'sample_data/mixed/sae.yaml'
    compute_sae(dataset_path, output_path)


if __name__ == '__main__':
    main()

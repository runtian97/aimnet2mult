#!/usr/bin/env python
"""Scan an HDF5 dataset for data quality issues that cause NaN during training.

Usage:
    python scripts/check_dataset.py path/to/dataset.h5
    python scripts/check_dataset.py path/to/fid0.h5 path/to/fid1.h5
"""

import sys
import numpy as np
import h5py


def check_h5(path):
    print(f"\n{'='*70}")
    print(f"Checking: {path}")
    print(f"{'='*70}")

    issues = []
    total_molecules = 0

    with h5py.File(path, "r") as f:
        for grp_name in sorted(f.keys()):
            grp = f[grp_name]
            n_atoms = int(grp_name)
            keys = list(grp.keys())
            n_mol = grp[keys[0]].shape[0] if keys else 0
            total_molecules += n_mol

            print(f"\n  Group {grp_name}: {n_mol} molecules, {n_atoms} atoms")
            print(f"    Keys: {keys}")

            for key in keys:
                arr = grp[key][:]
                n_nan = np.isnan(arr).sum()
                n_inf = np.isinf(arr).sum()
                vmin = np.nanmin(arr) if arr.size > 0 else 0
                vmax = np.nanmax(arr) if arr.size > 0 else 0

                status = "OK"
                if n_nan > 0:
                    status = f"*** {n_nan} NaN ***"
                    issues.append(f"{grp_name}/{key}: {n_nan} NaN values")
                if n_inf > 0:
                    status = f"*** {n_inf} Inf ***"
                    issues.append(f"{grp_name}/{key}: {n_inf} Inf values")

                print(f"    {key:16s} shape={str(arr.shape):20s} "
                      f"dtype={str(arr.dtype):10s} "
                      f"min={vmin:12.4g}  max={vmax:12.4g}  {status}")

                # Key-specific checks
                if key == "coord":
                    # Check for very close atoms
                    for mol_idx in range(min(n_mol, 10000)):  # sample up to 10k
                        coords = arr[mol_idx]
                        # Pairwise distances
                        diff = coords[:, None, :] - coords[None, :, :]
                        dist = np.sqrt((diff ** 2).sum(-1))
                        np.fill_diagonal(dist, 999.0)
                        min_dist = dist.min()
                        if min_dist < 0.3:
                            issues.append(
                                f"{grp_name}/coord: mol {mol_idx} has atoms "
                                f"only {min_dist:.4f} A apart"
                            )
                            if len(issues) > 20:
                                break
                    # Check for extreme coordinates
                    if np.abs(arr).max() > 100:
                        issues.append(
                            f"{grp_name}/coord: extreme coordinates "
                            f"(max |coord| = {np.abs(arr).max():.1f} A)"
                        )

                elif key == "energy":
                    # Check for extreme energies (post-SAE should be small)
                    if np.abs(arr).max() > 1e6:
                        issues.append(
                            f"{grp_name}/energy: extreme values "
                            f"(max |E| = {np.abs(arr).max():.4g})"
                        )
                    # Check for constant energy (all same value)
                    if arr.std() < 1e-10 and n_mol > 1:
                        issues.append(
                            f"{grp_name}/energy: all values identical "
                            f"(E = {arr[0]:.6g})"
                        )

                elif key == "forces":
                    if np.abs(arr).max() > 100:
                        issues.append(
                            f"{grp_name}/forces: extreme forces "
                            f"(max |F| = {np.abs(arr).max():.4g} eV/A)"
                        )

                elif key == "numbers":
                    unique = np.unique(arr)
                    print(f"      Elements present: {unique.tolist()}")
                    if 0 in unique and arr.ndim == 2:
                        # Count padding atoms per molecule
                        n_padding = (arr == 0).sum(axis=1)
                        if n_padding.max() > 0:
                            print(f"      Padding atoms: max {n_padding.max()} per molecule")

                elif key == "charge":
                    unique_charges = np.unique(arr)
                    print(f"      Unique charges: {unique_charges.tolist()}")

                elif key == "mult":
                    unique_mult = np.unique(arr)
                    print(f"      Unique multiplicities: {unique_mult.tolist()}")

    print(f"\n  Total molecules: {total_molecules}")
    print(f"\n{'='*70}")
    if issues:
        print(f"  ISSUES FOUND ({len(issues)}):")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  No issues found.")
    print(f"{'='*70}\n")

    return issues


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <dataset.h5> [dataset2.h5 ...]")
        sys.exit(1)

    all_issues = {}
    for path in sys.argv[1:]:
        all_issues[path] = check_h5(path)

    if any(all_issues.values()):
        print("\nSUMMARY: Data quality issues detected!")
        for path, issues in all_issues.items():
            if issues:
                print(f"  {path}: {len(issues)} issue(s)")
        sys.exit(1)
    else:
        print("\nSUMMARY: All datasets look clean.")

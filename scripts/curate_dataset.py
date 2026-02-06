#!/usr/bin/env python
"""Curate an HDF5 dataset by removing outlier molecules that cause NaN during training.

Filters applied per molecule (configurable via CLI flags):
  1. NaN / Inf in any field
  2. Extreme coordinates:  max |coord| > threshold  (default 100 A)
  3. Extreme energies:     |energy| > threshold      (default 1e6 eV)
  4. Extreme forces:       max |force| > threshold   (default 50 eV/A)
  5. Very close atoms:     min pairwise dist < threshold (default 0.3 A)

Usage:
    python scripts/curate_dataset.py input.h5 output.h5
    python scripts/curate_dataset.py input.h5 output.h5 --max-coord 80 --max-energy 5e5
    python scripts/curate_dataset.py input.h5 output.h5 --min-dist 0.5 --max-force 30
    python scripts/curate_dataset.py input.h5 output.h5 --dry-run   # report only, no output
"""

import argparse
import sys
import time

import h5py
import numpy as np


def compute_min_pairwise_dist(coords: np.ndarray, numbers: np.ndarray = None) -> np.ndarray:
    """Compute minimum pairwise distance for each molecule.

    Args:
        coords: (N, n_atoms, 3) array of coordinates.
        numbers: (N, n_atoms) array of atomic numbers. If provided, padding
                 atoms (Z=0) are excluded from the distance calculation.

    Returns:
        (N,) array of minimum pairwise distances per molecule.
    """
    n_mol, n_atoms, _ = coords.shape
    min_dists = np.full(n_mol, np.inf)

    # Process in chunks to avoid memory explosion for large groups
    chunk_size = max(1, min(2000, 500_000_000 // (n_atoms * n_atoms * 3 * 4)))
    for start in range(0, n_mol, chunk_size):
        end = min(start + chunk_size, n_mol)
        c = coords[start:end]  # (chunk, n_atoms, 3)
        diff = c[:, :, None, :] - c[:, None, :, :]  # (chunk, n_atoms, n_atoms, 3)
        dist2 = (diff ** 2).sum(-1)  # (chunk, n_atoms, n_atoms)

        # Mask diagonal (self-distances)
        eye = np.eye(n_atoms, dtype=bool)
        dist2[:, eye] = np.inf

        # Mask padding atoms (Z=0) if numbers are available
        if numbers is not None:
            nums = numbers[start:end]  # (chunk, n_atoms)
            pad_i = (nums == 0)  # (chunk, n_atoms)
            # If atom i or atom j is padding, set dist to inf
            pad_ij = pad_i[:, :, None] | pad_i[:, None, :]  # (chunk, n_atoms, n_atoms)
            dist2[pad_ij] = np.inf

        min_dists[start:end] = np.sqrt(dist2.reshape(end - start, -1).min(axis=-1))

    return min_dists


def filter_group(grp: h5py.Group, args) -> np.ndarray:
    """Return a boolean mask of molecules to KEEP for a given size group.

    Args:
        grp: HDF5 group for one atom-count bucket.
        args: Parsed CLI arguments with filter thresholds.

    Returns:
        (N,) boolean array — True = keep, False = remove.
    """
    keys = list(grp.keys())
    n_mol = grp[keys[0]].shape[0] if keys else 0
    keep = np.ones(n_mol, dtype=bool)

    # --- Filter 1: NaN / Inf in any float field ---
    for key in keys:
        arr = grp[key][:]
        if np.issubdtype(arr.dtype, np.floating):
            bad = np.isnan(arr) | np.isinf(arr)
            # Reduce to per-molecule
            while bad.ndim > 1:
                bad = bad.any(axis=-1)
            keep &= ~bad

    # --- Filter 2: Extreme coordinates ---
    if "coord" in grp and args.max_coord is not None:
        coord = grp["coord"][:]
        max_abs_coord = np.abs(coord).reshape(n_mol, -1).max(axis=-1)
        keep &= max_abs_coord <= args.max_coord

    # --- Filter 3: Extreme energies ---
    if "energy" in grp and args.max_energy is not None:
        energy = grp["energy"][:]
        keep &= np.abs(energy) <= args.max_energy

    # --- Filter 4: Extreme forces ---
    if "forces" in grp and args.max_force is not None:
        forces = grp["forces"][:]
        max_abs_force = np.abs(forces).reshape(n_mol, -1).max(axis=-1)
        keep &= max_abs_force <= args.max_force

    # --- Filter 5: Very close atoms ---
    if "coord" in grp and args.min_dist is not None:
        coord = grp["coord"][:]
        numbers = grp["numbers"][:] if "numbers" in grp else None
        min_dists = compute_min_pairwise_dist(coord, numbers)
        keep &= min_dists >= args.min_dist

    return keep


def curate_h5(input_path: str, output_path: str, args):
    """Read input H5, filter molecules, write clean output H5.

    Args:
        input_path: Path to input HDF5 dataset.
        output_path: Path to output HDF5 dataset (ignored if --dry-run).
        args: Parsed CLI arguments.
    """
    t0 = time.time()
    total_in = 0
    total_kept = 0
    group_stats = []

    out_f = None
    if not args.dry_run:
        out_f = h5py.File(output_path, "w")

    try:
        with h5py.File(input_path, "r") as f:
            group_names = sorted(f.keys(), key=lambda x: int(x))
            for grp_name in group_names:
                grp = f[grp_name]
                keys = list(grp.keys())
                n_mol = grp[keys[0]].shape[0] if keys else 0

                print(f"  Group {grp_name:>4s} ({n_mol:>10,d} mol): ", end="", flush=True)

                keep = filter_group(grp, args)
                n_kept = int(keep.sum())
                n_removed = n_mol - n_kept
                total_in += n_mol
                total_kept += n_kept
                pct = 100.0 * n_removed / n_mol if n_mol > 0 else 0.0

                print(f"kept {n_kept:>10,d}, removed {n_removed:>8,d} ({pct:5.2f}%)")

                group_stats.append((grp_name, n_mol, n_kept, n_removed))

                # Skip empty groups
                if n_kept == 0:
                    continue

                # Write filtered data
                if out_f is not None:
                    out_grp = out_f.create_group(grp_name)
                    for key in keys:
                        arr = grp[key][:]
                        out_grp.create_dataset(key, data=arr[keep], compression="gzip",
                                               compression_opts=1)

    finally:
        if out_f is not None:
            out_f.close()

    elapsed = time.time() - t0
    total_removed = total_in - total_kept
    pct_total = 100.0 * total_removed / total_in if total_in > 0 else 0.0

    print(f"\n{'='*70}")
    print(f"  Input:   {input_path}")
    if not args.dry_run:
        print(f"  Output:  {output_path}")
    print(f"  Total molecules in:  {total_in:>12,d}")
    print(f"  Total molecules out: {total_kept:>12,d}")
    print(f"  Removed:             {total_removed:>12,d}  ({pct_total:.3f}%)")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*70}")

    # Print groups with most removals
    group_stats.sort(key=lambda x: x[3], reverse=True)
    print(f"\n  Top groups by removals:")
    for grp_name, n_in, n_kept, n_removed in group_stats[:15]:
        if n_removed == 0:
            break
        pct = 100.0 * n_removed / n_in if n_in > 0 else 0.0
        print(f"    Group {grp_name:>4s}: {n_removed:>8,d} removed / {n_in:>10,d} ({pct:.2f}%)")

    print()
    return total_removed


def main():
    parser = argparse.ArgumentParser(
        description="Curate HDF5 dataset by removing outlier molecules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default thresholds
  python scripts/curate_dataset.py data/raw.h5 data/clean.h5

  # Stricter coordinate filter
  python scripts/curate_dataset.py data/raw.h5 data/clean.h5 --max-coord 80

  # Preview what would be removed (no output written)
  python scripts/curate_dataset.py data/raw.h5 --dry-run

  # Disable a specific filter by setting to 0
  python scripts/curate_dataset.py data/raw.h5 data/clean.h5 --min-dist 0
""",
    )
    parser.add_argument("input", help="Input HDF5 dataset path")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output HDF5 dataset path (required unless --dry-run)")
    parser.add_argument("--max-coord", type=float, default=100.0,
                        help="Remove molecules with |coord| > threshold (A). "
                             "Set to 0 to disable. Default: 100")
    parser.add_argument("--max-energy", type=float, default=1e6,
                        help="Remove molecules with |energy| > threshold (eV). "
                             "Set to 0 to disable. Default: 1e6")
    parser.add_argument("--max-force", type=float, default=50.0,
                        help="Remove molecules with max |force| > threshold (eV/A). "
                             "Set to 0 to disable. Default: 50")
    parser.add_argument("--min-dist", type=float, default=0.3,
                        help="Remove molecules with min pairwise distance < threshold (A). "
                             "Set to 0 to disable. Default: 0.3")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report what would be removed; don't write output.")

    args = parser.parse_args()

    # Convert 0 → None to disable filters
    if args.max_coord == 0:
        args.max_coord = None
    if args.max_energy == 0:
        args.max_energy = None
    if args.max_force == 0:
        args.max_force = None
    if args.min_dist == 0:
        args.min_dist = None

    if not args.dry_run and args.output is None:
        parser.error("output path is required unless --dry-run is specified")

    print(f"\n{'='*70}")
    print(f"Curating: {args.input}")
    print(f"Filters:")
    print(f"  max |coord|       : {args.max_coord if args.max_coord else 'disabled'} A")
    print(f"  max |energy|      : {args.max_energy if args.max_energy else 'disabled'} eV")
    print(f"  max |force|       : {args.max_force if args.max_force else 'disabled'} eV/A")
    print(f"  min pairwise dist : {args.min_dist if args.min_dist else 'disabled'} A")
    print(f"  NaN/Inf removal   : always on")
    print(f"{'='*70}\n")

    n_removed = curate_h5(args.input, args.output, args)

    if n_removed > 0 and args.dry_run:
        print("(Dry run — no output written. Re-run without --dry-run to produce clean dataset.)\n")


if __name__ == "__main__":
    main()

"""Physical unit conversion constants used across training and inference.

Dataset units (matching aimnet2):
- Energy: eV
- Forces: eV/Angstrom
- Distances: Angstrom
- Charges: elementary charge (e)

Datasets should be prepared with energies in eV and forces in eV/Å.
No automatic conversion from Hartree is performed during data loading.
"""

# Conversion constants from atomic units (for reference/external use)
HARTREE_TO_EV = 27.211386245988  # 1 Hartree = 27.2114 eV
BOHR_TO_ANGSTROM = 0.529177210903  # 1 Bohr = 0.5292 Angstrom
FORCE_HARTREE_BOHR_TO_EV_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM  # ~51.422 eV/Å per Ha/Bohr

# Conversion to display units (kcal/mol) for metrics
EV_TO_KCAL_MOL = 23.06054783061903  # 1 eV = 23.06 kcal/mol
FORCE_EV_ANGSTROM_TO_KCAL_MOL_ANGSTROM = EV_TO_KCAL_MOL  # Same factor for forces

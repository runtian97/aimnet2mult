"""Utilities exposed by the aimnet2mult package."""

from .units import HARTREE_TO_EV, FORCE_HARTREE_BOHR_TO_EV_ANGSTROM

# Note: Eager mode derivatives removed - use jit_wrapper for JIT models
# from aimnet2mult.models.jit_wrapper import load_jit_model

__all__ = [
    "HARTREE_TO_EV",
    "FORCE_HARTREE_BOHR_TO_EV_ANGSTROM",
]

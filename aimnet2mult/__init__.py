"""AIMNet2 multi-fidelity training package - Consolidated Independent Version"""

from . import data, models, train, ops, nbops, constants, modules, config, aev

__all__ = [
    "data",
    "models",
    "train",
    "ops",
    "nbops",
    "constants",
    "modules",
    "config",
    "aev",
    "__version__"
]
__version__ = "0.1.0"

"""Utility scripts for AIMNet2 multi-fidelity workflows."""

from importlib import import_module
from typing import Any

__all__ = ["compile_jit"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

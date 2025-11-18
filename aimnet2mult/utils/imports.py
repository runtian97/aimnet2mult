"""Utilities that ensure local AIMNet2 sources are importable."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

__all__ = ["ensure_aimnet2_on_path"]


    """
    Insert the sibling ``aimnet2`` repository into ``sys.path`` if needed.
    """
    current = Path(__file__).resolve()
    parents = current.parents

    candidates = []
    if len(parents) >= 3:
        candidates.append(parents[3] / "aimnet2")
    if len(parents) >= 2:
        candidates.append(parents[2] / "aimnet2")

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate_str

    return None

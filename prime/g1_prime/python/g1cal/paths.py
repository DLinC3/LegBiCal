"""Repository-root resolution and path policy.

Runtime data/model paths resolve inside the repository root so generated
artifacts and frozen inputs stay self-contained and relocatable.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_KEY = "G1CAL_ROOT"


def project_root() -> Path:
    """Return the repository root.

    Prefers ``G1CAL_ROOT``; falls back to walking up from this file. The
    result must contain the model manifest.
    """
    env = os.environ.get(_ENV_KEY)
    root = Path(env).resolve() if env else Path(__file__).resolve().parents[2]
    if not (root / "models/MODEL_MANIFEST.json").is_file():
        raise RuntimeError(f"not a G1 calibration root: {root}")
    return root


def resolve_inside_root(relative: str | Path, *, must_exist: bool = True) -> Path:
    """Resolve a repo-relative path, rejecting escapes from the root."""
    root = project_root()
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"absolute runtime paths are forbidden: {candidate}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes repository root: {candidate} -> {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved

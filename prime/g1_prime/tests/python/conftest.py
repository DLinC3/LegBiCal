"""Shared test utilities: per-run scratch directories under out/."""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from g1cal.paths import project_root


@pytest.fixture()
def fresh_scratch():
    """Return a function that recreates a test-owned scratch directory."""

    def recreate(relative: str) -> Path:
        if not relative.startswith("out/test_scratch/"):
            raise ValueError(
                f"{relative} is not a registered test scratch directory"
            )
        target = project_root() / relative
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        return target

    return recreate

"""CLI defaults and calibrated precision input."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from g1cal.calibration import calibrated_theta
from g1cal.cli import _demo, _resolve_theta


def test_precision_path_resolves_exact_calibrated_theta():
    resolved = _resolve_theta("data/calibrated/precision.csv")
    assert np.array_equal(resolved, calibrated_theta())


def test_demo_defaults_to_shipped_solutions(monkeypatch, tmp_path):
    calls = []

    def fake_render(clip, *, result_dir, output_root):
        calls.append((clip, result_dir, output_root))
        return {
            "se3_log_loss": 0.01,
            "artifacts": {
                "meshcat_html": {"path": f"{clip}.html"},
            },
        }

    monkeypatch.setattr("g1cal.rendering.render_clip", fake_render)
    args = SimpleNamespace(solve=False, out="out/test_scratch/demo")
    assert _demo(args) == 0
    assert [item[0] for item in calls] == ["run1", "run2"]
    assert all(item[1] is None for item in calls)

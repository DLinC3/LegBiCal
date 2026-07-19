"""Stage 3 covariance units, positivity, whitening, and derivatives."""

import numpy as np
import pytest

from g1cal.covariance import (
    BLOCKS,
    FLOOR_RATIO,
    CovarianceParameterization,
)


def test_locked_block_layout_and_dimensions():
    p = CovarianceParameterization()
    assert p.size == 17
    assert [b.matrix for b in BLOCKS].count("P0") == 7
    assert [b.matrix for b in BLOCKS].count("Q") == 3
    assert [b.matrix for b in BLOCKS].count("R") == 7
    c = p.evaluate(np.zeros(17))
    assert c.variance_diag["P0"].shape == (70,)
    assert c.variance_diag["Q"].shape == (35,)
    assert c.variance_diag["R"].shape == (70,)


def test_positive_covariance_precision_and_whitening_identity():
    p = CovarianceParameterization()
    c = p.evaluate(np.linspace(-2.5, 2.5, 17))
    for matrix in ("P0", "Q", "R"):
        assert np.all(c.variance_diag[matrix] > 0)
        assert np.allclose(
            c.variance_diag[matrix] * c.precision_diag[matrix], 1.0
        )
        assert np.allclose(
            c.whitening_diag[matrix] ** 2, c.precision_diag[matrix]
        )


def test_reference_sigma_and_unit_specific_floor():
    p = CovarianceParameterization()
    c = p.evaluate(np.zeros(17))
    for block in BLOCKS:
        expected = block.sigma_ref * np.sqrt(1.0 + FLOOR_RATIO**2)
        assert c.sigma_by_block[block.name] == pytest.approx(expected)


def test_bounds_and_shape_are_enforced():
    p = CovarianceParameterization()
    with pytest.raises(ValueError):
        p.evaluate(np.zeros(16))
    theta = np.zeros(17)
    theta[3] = 3.01
    with pytest.raises(ValueError):
        p.evaluate(theta)


def test_precision_jacobian_matches_central_difference():
    p = CovarianceParameterization()
    theta = np.linspace(-0.5, 0.5, 17)
    base = p.evaluate(theta)
    h = 1e-6
    for index in (0, 7, 12, 16):
        plus = theta.copy(); plus[index] += h
        minus = theta.copy(); minus[index] -= h
        cp, cm = p.evaluate(plus), p.evaluate(minus)
        for matrix in ("P0", "Q", "R"):
            fd = (cp.precision_diag[matrix] - cm.precision_diag[matrix]) / (2 * h)
            analytic = base.precision_jacobian[matrix][:, index]
            scale = max(1.0, float(np.max(np.abs(fd))))
            assert np.max(np.abs(fd - analytic)) / scale < 1e-8

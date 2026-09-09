"""Tests for initial data construction."""

import numpy as np
import pytest

from pbh.initial import compute_deltam0, growingmode, makegrid


def test_uniform_grid_has_no_origin_or_endpoint() -> None:
    grid = makegrid(gridpoints=10, squeeze=0, Amax=5)
    assert len(grid) == 10
    assert grid[0] == pytest.approx(0.25)
    assert grid[-1] == pytest.approx(4.75)
    assert np.diff(grid) == pytest.approx(0.5)


def test_squeezed_grid_is_denser_near_origin() -> None:
    grid = makegrid(gridpoints=100, squeeze=2, Amax=10)
    assert len(grid) == 100
    assert np.all(grid > 0)
    assert np.all(grid < 10)
    assert np.all(np.diff(grid) > 0)
    spacing = np.diff(grid)
    assert spacing[0] < spacing[-1]


def test_deltam0_is_gaussian() -> None:
    grid = makegrid(gridpoints=50, squeeze=0, Amax=10)
    deltam0 = compute_deltam0(grid, amplitude=0.1, sigma=2)
    assert deltam0 == pytest.approx(0.1 * np.exp(-(grid**2) / 8))
    assert np.all(np.diff(deltam0) < 0)


def test_growing_mode_reduces_to_background_for_zero_perturbation() -> None:
    grid = makegrid(gridpoints=50, squeeze=0, Amax=10)
    r, u, m = growingmode(grid, np.zeros_like(grid))
    assert r == pytest.approx(grid)
    assert u == pytest.approx(grid)
    assert m == pytest.approx(1.0)


def test_growing_mode_is_overdense_at_centre() -> None:
    grid = makegrid(gridpoints=200, squeeze=2, Amax=10)
    r, u, m = growingmode(grid, compute_deltam0(grid, amplitude=0.1))
    assert r.shape == u.shape == m.shape == grid.shape
    assert m[0] > 1.0
    # Far from the perturbation we recover the background
    assert m[-1] == pytest.approx(1.0, abs=1e-4)
    assert u[-1] / r[-1] == pytest.approx(1.0, abs=1e-4)
    # An overdensity contracts relative to the background
    assert u[0] / r[0] < 1.0

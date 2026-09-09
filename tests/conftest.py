"""Shared fixtures for the pbh test suite."""

import numpy as np
import pytest

from pbh.base import FloatArray
from pbh.initial import compute_deltam0, growingmode, makegrid


@pytest.fixture
def small_initial_data() -> tuple[FloatArray, FloatArray, FloatArray]:
    """Growing-mode initial data on a small squeezed grid (fast to evolve)."""
    grid = makegrid(gridpoints=150, squeeze=2, Amax=10)
    return growingmode(grid, compute_deltam0(grid, amplitude=0.175))


@pytest.fixture
def random_grid() -> FloatArray:
    """A sorted random grid of positive points in (0, 2pi)."""
    rng = np.random.default_rng(12345)
    return np.sort(rng.uniform(0.05, 2 * np.pi, size=40))

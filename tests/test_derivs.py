"""Tests for finite difference derivatives."""

import numpy as np
import pytest

from pbh.base import FloatArray
from pbh.derivs import Derivative, DerivativeError


def uniform_grid(n: int, xmax: float = np.pi) -> FloatArray:
    """Uniform grid with no point at the origin (first point at half a spacing)."""
    h = xmax / n
    return np.arange(h / 2, xmax, h)


def test_even_derivative_of_cosine(random_grid: FloatArray) -> None:
    diff = Derivative(random_grid)
    result = diff.dydx(np.cos(random_grid), even=True)
    assert result == pytest.approx(-np.sin(random_grid), abs=0.15)


def test_odd_derivative_of_sine(random_grid: FloatArray) -> None:
    diff = Derivative(random_grid)
    result = diff.dydx(np.sin(random_grid), even=False)
    assert result == pytest.approx(np.cos(random_grid), abs=0.15)


def test_rhoderiv_matches_analytic(random_grid: FloatArray) -> None:
    diff = Derivative(random_grid)
    y = np.cos(random_grid)
    expected = (-random_grid * y - 4 * np.sin(random_grid)) / 3
    result = diff.rhoderiv(y)
    # Last point is not computed
    assert result[-1] == 0
    assert result[:-1] == pytest.approx(expected[:-1], abs=0.2)


def test_rhoderiv_lagrange_matches_stencil(random_grid: FloatArray) -> None:
    diff = Derivative(random_grid)
    y = np.cos(random_grid)
    stencil = diff.rhoderiv(y)
    lagrange = Derivative.rhoderiv_lagrange(y, random_grid)
    assert lagrange == pytest.approx(stencil, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("even", [True, False])
def test_dydx_is_second_order_accurate(even: bool) -> None:
    """Halving the grid spacing should reduce the error by about a factor of four."""

    def dfunc(x: FloatArray) -> FloatArray:
        return -np.sin(x) if even else np.cos(x)

    func = np.cos if even else np.sin
    errors: list[float] = []
    for n in (50, 100, 200):
        x = uniform_grid(n)
        result = Derivative(x).dydx(func(x), even=even)
        errors.append(float(np.max(np.abs(result - dfunc(x)))))
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all(ratios > 3.5)


def test_rhoderiv_convergence() -> None:
    """Document the convergence of the rho derivative operator on a uniform grid.

    At fixed position x, the operator is second order accurate, but the error coefficient grows like h^2/x towards
    the origin, because x^4 is not well approximated by its cell average when x is comparable to the grid spacing.
    Consequently the gridpoints immediately adjacent to the origin (index 1, 2, ...) only converge at first order.
    The innermost point uses a dedicated formula and is unaffected.
    """
    errors: list[FloatArray] = []
    for n in (50, 100, 200):
        x = uniform_grid(n)
        y = np.cos(x)
        expected = (-x * y - 4 * np.sin(x)) / 3
        err = np.abs(Derivative(x).rhoderiv(y) - expected)
        errors.append(np.array([err[0], err[1], err[(x > 1.0)][:-1].max()]))
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    # Innermost point: at least second order
    assert np.all(ratios[:, 0] > 3.5)
    # Second point: only first order
    assert np.all((ratios[:, 1] > 1.8) & (ratios[:, 1] < 2.5))
    # Interior (x > 1): second order
    assert np.all(ratios[:, 2] > 3.5)


def test_derivative_of_constant_is_zero(random_grid: FloatArray) -> None:
    diff = Derivative(random_grid)
    ones = np.ones_like(random_grid)
    assert diff.dydx(ones, even=True) == pytest.approx(0, abs=1e-12)
    # Large cancellations in the x^4-weighted stencil limit the achievable precision here
    assert diff.rhoderiv(ones) == pytest.approx(0, abs=1e-9)


def test_derivative_of_linear_odd_function_is_exact(random_grid: FloatArray) -> None:
    diff = Derivative(random_grid)
    assert diff.dydx(3 * random_grid, even=False) == pytest.approx(3, rel=1e-10)


def test_grid_too_short() -> None:
    with pytest.raises(DerivativeError, match="too short"):
        Derivative(np.array([0.5, 1.5]))


def test_mismatched_lengths(random_grid: FloatArray) -> None:
    diff = Derivative(random_grid)
    with pytest.raises(DerivativeError, match="different dimensions"):
        diff.dydx(np.ones(len(random_grid) + 1), even=True)

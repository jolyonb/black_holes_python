"""Spectral properties of the assembled finite difference operators.

The instability that plagued the Eulerian code is an odd-even ("sawtooth") grid mode. The centred first-derivative
stencil maps the alternating vector (-1)^i to (almost) zero, so that mode is invisible to dmdr, dudr, the advection
terms and the artificial viscosity trigger, and nothing damps it. By contrast the flux-form rhoderiv operator,
composed with 1/r as it appears in the u equation, is symmetric in a weighted inner product with a purely real,
negative spectrum: it is not the problem. These tests pin both facts to the actual sparse matrices.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from pbh.derivs import Derivative
from pbh.initial import makegrid

GRIDS = [
    pytest.param(600, 2.0, id="default-sinh-600"),
    pytest.param(600, 0.0, id="uniform-600"),
    pytest.param(150, 2.0, id="coarse-sinh-150"),
]


@pytest.mark.parametrize(("gridpoints", "squeeze"), GRIDS)
def test_rho_operator_over_r_has_real_negative_spectrum(gridpoints: int, squeeze: float) -> None:
    """(1/r) rhoderiv is the discrete (1/3) r^-4 d/dr (r^4 d/dr).

    In the continuum this is self-adjoint and negative in the r^4 dr inner product. The discrete analogue of the
    weight for row i is r_i (r_{i+1}^4 - r_{i-1}^4). This is the operator that carries the sound waves in the u
    equation, so its eigenvalues must be real and negative.
    """
    r = makegrid(gridpoints=gridpoints, squeeze=squeeze, Amax=10)
    n = len(r)
    rho = Derivative(r).rho_matrix.toarray()
    # Drop the empty last row (the boundary condition replaces it) and its column
    operator = (np.diag(1.0 / r) @ rho)[:-1, :-1]
    eigenvalues = np.linalg.eigvals(operator)
    scale = np.abs(eigenvalues).max()
    assert np.abs(eigenvalues.imag).max() <= 1e-9 * scale  # purely real
    assert eigenvalues.real.max() < 0  # strictly negative (no zero mode: the origin closure removes it)

    # Weighted symmetry in the interior. The origin row uses a special closure and is only approximately symmetric
    # (relative asymmetry about 8e-3 on every grid), so the first rows are excluded.
    r4 = r**4
    weight = np.empty(n)
    weight[1:-1] = r[1:-1] * (r4[2:] - r4[:-2])
    weight[0] = r[0] * r4[1]
    weight[-1] = weight[-2]
    symmetrised = (np.diag(weight) @ (np.diag(1.0 / r) @ rho))[:-1, :-1]
    interior = slice(2, n - 3)
    block = symmetrised[interior, interior]
    asymmetry = np.abs(block - block.T).max() / np.abs(block).max()
    assert asymmetry < 1e-8


def test_centred_first_derivative_annihilates_sawtooth() -> None:
    """CHARACTERISATION OF A KNOWN DEFECT.

    The collocated centred difference has the grid-scale alternating mode in its null space (exactly, on a uniform
    grid). This is why odd-even noise in m and u is neither advected nor damped nor seen by the viscosity trigger
    Q ~ (dudr)^2. Expected to change when a staggered layout replaces dydx; at that point invert or delete this test.
    On the sinh grid the residual is grid dependent (about 0.36 against 2/h of 218), so only the uniform grid is
    asserted.
    """
    r = makegrid(gridpoints=600, squeeze=0, Amax=10)
    sawtooth = (-1.0) ** np.arange(len(r))
    h = np.diff(r).min()
    result = Derivative(r).even_matrix @ sawtooth
    assert np.abs(result[2:-2]).max() < 1e-8 * (2.0 / h)  # the true |d(sawtooth)/dr| would be about 2/h


def linearised_system(r: NDArray[np.float64], xi: float) -> NDArray[np.float64]:
    """Assemble the semi-discrete linearised Misner-Sharp system for (dm, dU) at fixed xi, w = 1/3.

    Interior rows: d_xi dm = dm/2 - 2 dU, d_xi dU = -dm/4 - (e^xi / 8) (1/r) rhoderiv(dm).
    The last dU row is the outgoing-wave boundary condition, Eqs. (101a-d) and (102), with derivatives taken from
    the last row of the even first-derivative operator (the linear dU is even).
    """
    n = len(r)
    diff = Derivative(r)
    identity = np.eye(n)
    rho = diff.rho_matrix.toarray()
    d_dr_last = diff.even_matrix.toarray()[-1]

    dm_dot = np.hstack([0.5 * identity, -2 * identity])
    du_dot = np.hstack([-0.25 * identity - np.exp(xi) / 8 * (np.diag(1.0 / r) @ rho), np.zeros((n, n))])

    # Boundary row
    c = np.exp(xi / 2) / np.sqrt(12)
    R = r[-1]
    den = R * (2 * c + R)
    W = -c
    X = -(12 * c**2 + 6 * c * R + R**2) / (2 * den)
    Y = c * (2 * c**2 + 2 * c * R + R**2) / (2 * den)
    Z = c * (3 * c**2 + 3 * c * R + R**2) / (R * den)
    boundary = np.zeros(2 * n)
    boundary[:n] = Y * d_dr_last
    boundary[n - 1] += Z
    boundary[n:] = W * d_dr_last
    boundary[2 * n - 1] += X
    du_dot[-1] = boundary

    return np.vstack([dm_dot, du_dot])


@pytest.mark.parametrize("xi", [0.0, 3.0, 6.0])
def test_linearised_system_has_no_spurious_growth(xi: float) -> None:
    """The linearised system including the boundary row has no eigenvalue growing faster than physics allows.

    The superhorizon growing mode has rate 1 (and less once inside the horizon); every oscillatory (sound wave)
    eigenvalue must have real part equal to the frozen-coefficient value 1/4. Anything above signals a spurious
    boundary or interior instability.
    """
    r = makegrid(gridpoints=300, squeeze=2, Amax=10)
    eigenvalues = np.linalg.eigvals(linearised_system(r, xi))
    assert eigenvalues.real.max() <= 1.0 + 1e-9
    oscillatory = eigenvalues[np.abs(eigenvalues.imag) > 5]
    assert len(oscillatory) > 0
    assert oscillatory.real.max() == pytest.approx(0.25, abs=1e-3)

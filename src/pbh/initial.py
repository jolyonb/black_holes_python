"""Initial data construction for Misner-Sharp evolution.

Provides helpers to construct a grid, an initial density perturbation, and the growing mode solution built from it.
"""

import numpy as np

from pbh.base import FloatArray
from pbh.derivs import Derivative


def makegrid(gridpoints: int, squeeze: float = 2, Amax: float = 14) -> FloatArray:
    """Create a grid with a given Amax value and the specified number of gridpoints.

    If squeeze is 0, the grid is even, else it's squeezed towards the origin.
    Note that there is no gridpoint at the origin or at Amax.
    """
    delta = Amax / gridpoints
    grid = np.arange(delta / 2, Amax, delta)
    if squeeze == 0:
        return grid
    return Amax * np.sinh(squeeze * grid / Amax) / np.sinh(squeeze)


def compute_deltam0(grid: FloatArray, amplitude: float = 0.17, sigma: float = 2) -> FloatArray:
    """Construct deltam as a Gaussian on the given grid.

    Note that the critical amplitude for black hole formation is around 0.1737 (for sigma=2).
    """
    return amplitude * np.exp(-grid * grid / 2 / sigma / sigma)


def growingmode(grid: FloatArray, deltam0: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute the growing mode based on deltam0 and a comoving grid.

    Returns:
        A tuple (r, u, m) of initial field values on the grid.
    """
    # Initialize a differentiator
    diff = Derivative(grid)

    # Compute dm
    dm = diff.dydx(deltam0, even=True)

    # Initial data
    deltam1 = 1.0 * deltam0
    deltau1 = -0.25 * deltam0
    deltarho1 = deltam0 + grid * dm / 3
    deltar1 = -1 / 8 * (deltam0 + deltarho1)

    ddeltarho1 = diff.dydx(deltarho1, even=True)
    ddeltar1 = diff.dydx(deltar1, even=True)

    deltam2 = (
        deltau1 / 5 * (2 * deltau1 - 6 * deltam1 - deltarho1)
        + deltarho1 / 40 * (10 * deltam1 - 3 * deltarho1)
        + ddeltarho1 / 10 / grid
    )
    ddeltam2 = diff.dydx(deltam2, even=True)
    deltau2 = (
        3 / 20 * (deltau1 * (deltam1 + deltarho1 - 2 * deltau1) - deltarho1 * deltarho1 / 4 - ddeltarho1 / 2 / grid)
    )
    deltarho2 = deltam2 + grid * (ddeltam2 / 3 - (deltarho1 - deltam1) * ddeltar1)
    deltar2 = (
        1 / 16 * (4 * deltar1 * deltau1 + 4 * deltau2 - deltarho2 + deltarho1 * (5 / 8 * deltarho1 - deltar1 - deltau1))
    )

    # Starting variables
    m = 1.0 + deltam1 + deltam2
    u = 1.0 + deltau1 + deltau2
    r = (1.0 + deltar1 + deltar2) * grid

    u *= r

    return r, u, m

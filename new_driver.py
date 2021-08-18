from baseclass import Evolver
from math import exp, pi, sqrt, log
import numpy as np
from newderivs import Derivative
from fancyderivs import Derivative as DerivativeFancy
from BlackHoleEvolver import Lagrangian


# Lagrangian w MS
# Set up our gridpoints in A

def makegrid(gridpoints, squeeze=2, Amax=14):
    """Creates a grid"""
    delta = Amax / gridpoints
    grid = np.arange(delta / 2, Amax, delta)
    if squeeze == 0:
        return grid
    grid = Amax * np.sinh(squeeze * grid / Amax) / np.sinh(squeeze)
    return grid


def compute_deltam0(grid, amplitude=0.17):
    """Constructs deltam0 based on a given grid"""
    sigma = 2
    # amplitude = 0.16  # 0.1737 < criticality in here somewhere? < 0.173711
    return amplitude * np.exp(- grid * grid / 2 / sigma / sigma)


def growingmode(grid, deltam0):
    """Computes the growing mode based on deltam0 and a comoving grid"""
    # Initialize a differentiator
    diff = DerivativeFancy(4)
    diff.set_x(grid, 1)

    # Compute dm
    dm = diff.dydx(deltam0)

    # Initial data
    deltam1 = 1.0 * deltam0
    deltau1 = - 0.25 * deltam0
    deltarho1 = deltam0 + grid * dm / 3
    deltar1 = -1 / 8 * (deltam0 + deltarho1)

    ddeltarho1 = diff.dydx(deltarho1)
    ddeltar1 = diff.dydx(deltar1)

    deltam2 = (deltau1 / 5 * (2 * deltau1 - 6 * deltam1 - deltarho1)
               + deltarho1 / 40 * (10 * deltam1 - 3 * deltarho1)
               + ddeltarho1 / 10 / grid)
    ddeltam2 = diff.dydx(deltam2)
    deltau2 = 3 / 20 * (deltau1 * (deltam1 + deltarho1 - 2 * deltau1)
                        - deltarho1 * deltarho1 / 4 - ddeltarho1 / 2 / grid)
    deltarho2 = deltam2 + grid * (ddeltam2 / 3 - (deltarho1 - deltam1) * ddeltar1)
    deltar2 = 1 / 16 * (4 * deltar1 * deltau1 + 4 * deltau2 - deltarho2
                        + deltarho1 * (5 / 8 * deltarho1 - deltar1 - deltau1))

    # Starting variables
    m = 1.0 + deltam1 + deltam2
    u = 1.0 + deltau1 + deltau2
    r = (1.0 + deltar1 + deltar2) * grid

    u *= r

    return r, u, m


grid = makegrid(500, squeeze=0, Amax=14)
deltam0 = compute_deltam0(grid, amplitude=0.120)
r, u, m = growingmode(grid, deltam0)
xi0 = 0.0
evol = Lagrangian(grid, r, u, m, xi0, debug=True)

evol.drive(1, open("boundary18.dat", "w"), 7, 0)
print("Done!")

"""
driver.py

Entry point for evolving black holes.
"""

import sys
import numpy as np

from ms import MS, MSEulerian, MSLagrangian
from derivs import Derivative


def makegrid(gridpoints: int, squeeze: float = 2, Amax: float = 14):
    """
    Creates a grid with a given Amax value and the specified number of gridpoints.
    If squeeze is 0, the grid is even, else it's squeezed towards the origin.
    Note that there is no gridpoint at the origin or at Amax.
    """
    delta = Amax / gridpoints
    grid = np.arange(delta / 2, Amax, delta)
    if squeeze == 0:
        return grid
    return Amax * np.sinh(squeeze * grid / Amax) / np.sinh(squeeze)


def compute_deltam0(grid: np.ndarray, amplitude: float = 0.17):
    """
    Constructs deltam as a Gaussian on the given grid.
    Note that the critical amplitude for black hole formation is around 0.1737 (for sigma=2).
    """
    sigma = 2
    return amplitude * np.exp(- grid * grid / 2 / sigma / sigma)


def growingmode(grid: np.ndarray, deltam0: np.ndarray):
    """Computes the growing mode based on deltam0 and a comoving grid"""
    # Initialize a differentiator
    diff = Derivative(grid)

    # Compute dm
    dm = diff.dydx(deltam0, even=True)

    # Initial data
    deltam1 = 1.0 * deltam0
    deltau1 = - 0.25 * deltam0
    deltarho1 = deltam0 + grid * dm / 3
    deltar1 = -1 / 8 * (deltam0 + deltarho1)

    ddeltarho1 = diff.dydx(deltarho1, even=True)
    ddeltar1 = diff.dydx(deltar1, even=True)

    deltam2 = (deltau1 / 5 * (2 * deltau1 - 6 * deltam1 - deltarho1)
               + deltarho1 / 40 * (10 * deltam1 - 3 * deltarho1)
               + ddeltarho1 / 10 / grid)
    ddeltam2 = diff.dydx(deltam2, even=True)
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

def main():
    # Set up our driver here
    driver = MS(black_hole_check=True,
                enforce_timeout=False,
                eomhandler=MSEulerian,  # MSEulerian or MSLagrangian
                viscosity=2,
                debug=True)

    # Construct initial conditions
    if len(sys.argv) == 2:
        # Load initial conditions from file
        driver.load_initial_conditions(sys.argv[1])
    else:
        # Initialize grid
        grid = makegrid(gridpoints=600, squeeze=2, Amax=10)
        # Initialize deltam0
        deltam0 = compute_deltam0(grid, amplitude=0.175)
        # Initialize r, u and m based on deltam0
        r, u, m = growingmode(grid, deltam0)
        # Starting time
        xi0 = 0.0
        # Initialize drivers
        driver.set_initial_conditions(xi0, r, u, m)

    # Run!
    print('Evolver initialized. Beginning evolution!')
    with open('output.dat', 'w') as f:
        driver.drive(output_step=0.1,
                     file_handle=f,
                     max_time=7,
                     write_after=0)
    print(f'Evolution complete! Status: {driver.status.name}')

if __name__ == '__main__':
    main()

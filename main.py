#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main routine that runs the program
"""
import sys
import numpy as np
from driver import Driver, Status
from fancyderivs import Derivative

def makegrid(gridpoints, squeeze=2):
    """Creates a grid"""
    Amax = 12
    delta = Amax / gridpoints
    grid = np.arange(delta / 2, Amax, delta)
    if squeeze == 0:
        return grid
    grid = Amax * np.sinh(squeeze * grid / Amax) / np.sinh(squeeze)
    return grid

def compute_deltam0(grid):
    """Constructs deltam0 based on a given grid"""
    sigma = 2
    amplitude = 0.174  # 0.1737 < criticality in here somewhere? < 0.173711
    return amplitude * np.exp(- grid * grid / 2 / sigma / sigma)

def growingmode(grid, deltam0):
    """Computes the growing mode based on deltam0 and a comoving grid"""
    # Initialize a differentiator
    diff = Derivative(4)
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

    deltam2 = (deltau1/5*(2*deltau1 - 6*deltam1 - deltarho1)
               + deltarho1/40*(10*deltam1 - 3*deltarho1)
               + ddeltarho1/10/grid)
    ddeltam2 = diff.dydx(deltam2)
    deltau2 = 3/20*(deltau1*(deltam1 + deltarho1 - 2*deltau1)
                    - deltarho1 * deltarho1 / 4 - ddeltarho1/2/grid)
    deltarho2 = deltam2 + grid * (ddeltam2/3 - (deltarho1-deltam1) * ddeltar1)
    deltar2 = 1/16*(4*deltar1*deltau1 + 4*deltau2 - deltarho2
                    + deltarho1*(5/8*deltarho1 - deltar1 - deltau1))

    # Starting variables
    m = 1.0 + deltam1 + deltam2
    u = 1.0 + deltau1 + deltau2
    r = (1.0 + deltar1 + deltar2) * grid

    return r, u, m

# Set up the output file
f = open("output.dat", "w")

# Make the grid and initial data
if len(sys.argv) == 1:
    grid = makegrid(800)
    deltam0 = compute_deltam0(grid)
    r, u, m = growingmode(grid, deltam0)
    xi0 = 0.0
else:
    # Read in from the data file
    print("Loading data from {}".format(sys.argv[1]))
    with open(sys.argv[1]) as infile:
        data = infile.readlines()
    xi0 = float(data[0])
    r = np.zeros(len(data) - 1)
    u = np.zeros(len(data) - 1)
    m = np.zeros(len(data) - 1)
    for idx, line in enumerate(data[1:]):
        rval, uval, mval = line.split("\t")
        r[idx] = float(rval)
        u[idx] = float(uval)
        m[idx] = float(mval)
    print("Grid size is {}".format(len(r)))
    print("Starting evolution at xi = {}".format(xi0))

# Construct the driver
mydriver = Driver(r, u, m, jumptime=0, debug=True, xi0=xi0)

# Start by performing the MS evolution
print("Beginning MS evolution")
mydriver.runMS(f, timestep=0.1)

# Check to see what our status is
if mydriver.status == Status.MS_IntegrationError:
    print("Unable to integrate further: {}".format(mydriver.msg))
    if mydriver.hit50:
        print("Central density hit 50x background density")
        if mydriver.stalled:
            print("However, it appears that the collapse stalled, with central density falling")
            print("Integration error likely due to shocks or bad derivatives")
elif mydriver.status == Status.MS_MaxTimeReached:
    print("Maximum time reached")
elif mydriver.status == Status.MS_NegativeDensity:
    print("Negative density detected")
elif mydriver.status == Status.BlackHoleFormed:
    print("Black hole detected!")

# Tidy up
f.close()
print("Done!")

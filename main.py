#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs stuff
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import ms, rb
from finitediff import Derivative

# Set up the data object
data = ms.Data()

# Set some parameters
data.gridpoints = 750
data.Amax = 20
data.bhcheck = True

# Initialise deltam spline
sigma = data.Amax / 10
initA = np.linspace(-1, data.Amax, 1000)
vals = 0.174 * np.exp(- initA * initA / 2 / sigma / sigma)
deltamspline = InterpolatedUnivariateSpline(initA, vals)

# Grid points in A
delta = data.Amax / data.gridpoints
gridi = np.arange(delta/2, data.Amax, delta)
#grid = data.Amax * np.sinh(4*gridi/data.Amax)/np.sinh(4)
grid = gridi

# deltam on grid
deltam = np.array([deltamspline(i) for i in grid])

# Initialize a differentiator
diff = Derivative(4)
diff.set_x(grid, 1)

# Compute dm
dm = diff.dydx(deltam)

# Initial data
deltam1 = 1.0 * deltam
deltau1 = - 0.25 * deltam
deltarho1 = deltam + grid * dm / 3
deltar1 = -1/8*(deltam + deltarho1)

ddeltarho1 = diff.dydx(deltarho1)
ddeltar1 = diff.dydx(deltar1)

deltam2 = deltau1/5*(2*deltau1 -6*deltam1 -deltarho1) + deltarho1/40*(10*deltam1-3*deltarho1) + ddeltarho1/10/grid
ddeltam2 = diff.dydx(deltam2)

deltau2 = 3/20*(deltau1 * (deltam1 + deltarho1 - 2 * deltau1) - deltarho1 * deltarho1 / 4 - ddeltarho1/2/grid)

deltarho2 = deltam2 + grid * (ddeltam2/3 - (deltarho1-deltam1) * ddeltar1)

deltar2 = 1/16*(4*deltar1 * deltau1 + 4*deltau2 - deltarho2 + deltarho1*(5/8*deltarho1 -deltar1 -deltau1))

# Starting variables
m = 1.0 + deltam1 + deltam2
u = 1.0 + deltau1 + deltau2
r = (1.0 + deltar1 + deltar2) * grid
data.umr = np.concatenate((u, m, r))

# How long we run
maxtime = 7
# How often we report
timestep = 0.05

# Initialize integrator
data.initialize(0.0)

# Output file
f = open("output.dat", "w")

# Begin evolution
print("Here we go!")
data.write_data(f)
#import sys
#sys.exit()
#data.step(4.0)
data.write_data(f)

print("First big step taken!")
while data.integrator.t < maxtime :
    newtime = data.integrator.t + timestep
    if newtime > maxtime :
        newtime = maxtime
    result = data.step(newtime)
    if result == -1 :
        print("Cannot integrate further")
        break
    # Successful step
    data.write_data(f)
    print("Done:", data.integrator.t)
    if result == 1 :
        break
#result = 1

# Check how we did
if result == 0 :
    print("Max time reached")
elif result == -1 :
    print("Integration unable to continue")
elif result == 1 :
    print("Black hole detected!")
    # Now start the next phase
    newdata = rb.Data()
    newdata.umrrho = np.concatenate((data.u, data.m, data.r, data.rho))
    newdata.initialize(data.integrator.t)

    # Set up starting tau and width for transition
    
    # Find sound horizon
    sound = data.r[np.where(data.csp < 0)][-1]
#    print(sound)
    newdata.A1 = sound
    newdata.A0 = 0.0
    newdata.xi0 = data.integrator.t

    maxtime = 7.0

    while newdata.integrator.t < maxtime :
        newtime = newdata.integrator.t + timestep
        if newtime > maxtime :
            newtime = maxtime
        result = newdata.step(newtime)
        if result == -1 :
            print("Cannot integrate further")
            break
        # Successful step
        newdata.write_data(f)
        print("NewDone:", newdata.integrator.t)

# Tidy up
f.close()
print("Done!")
print('To plot, use gnuplot, like: plot "output.dat" i 1:50:1 u 1:(2*$7/$8) w l')
# Also helpful to know "set log y"

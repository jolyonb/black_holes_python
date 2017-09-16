#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main routine that runs the program
"""

import numpy as np
from driver import Driver

# Set up our gridpoints in A
gridpoints = 500
Amax = 14
delta = Amax / gridpoints
grid = np.arange(delta/2, Amax, delta)
#grid = Amax * np.sinh(4*gridi/Amax)/np.sinh(4)

# Compute deltam0 on our grid
sigma = 2
amplitude = 0.1737 # 0.1737 < criticality in here somewhere? < 0.173711
deltam0 = amplitude * np.exp(- grid * grid / 2 / sigma / sigma)

# Set up the output file
f = open("output.dat", "w")

# Set up the driver
mydriver = Driver(MSfile=f,
                  RBfile=f,
                  maxtime=6.5,
                  timestep=0.05,
                  bhcheck=True,
                  runRB=False,
                  jumptime=5.0)
mydriver.construct_init_data(deltam0, grid)

# Run everything
mydriver.run()

# Tidy up
f.close()
print("Done!")

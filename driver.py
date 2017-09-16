#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Containts the class that drives the evolution
"""
from enum import Enum
import numpy as np
import ms, rb
from ms import IntegrationError, BlackHoleFormed, ShellCrossing
from fancyderivs import Derivative

class Status(Enum):
    OK = 0
    BlackHoleFormed = 1
    MassExtracted = 2
    MS_MaxTimeReached = 3
    RB_MaxTimeReached = 4
    MS_IntegrationError = -1
    RB_IntegrationError = -2
    MS_ShellCrossing = -3
    RB_ShellCrossing = -4

class Driver(object):
    """
    Sets up and runs the entire black hole evolution
    Order of operations:
    driver = Driver(MSfile, RBfile)
    driver.construct_init_data(deltam0, grid)
    driver.run()
    Resulting status is stored in driver.status
    """

    def __init__(self, MSfile, RBfile,
                 maxtime=7, MStimestep=0.1, RBtimestep=0.01, bhcheck=True, runRB=True, jumptime=4.0):
        """
        Set parameters for driving this run
        MSfile - file handle to write to in MS evolution
        RBfile - file handle to write to in RB evolution (can be same as MSfile)
        maxtime - Final time to run to
        MStimestep - Step size at which we output data to file (MS)
        RBtimestep - Step size at which we output data to file (RB)
        bhcheck - Controls whether or not we check for black holes in MS
        runRB - Controls whether or not to go into RB evolution upon detection of a black hole
        jumptime - Time before which no writes take place
        """
        # Initialize variables
        self.status = Status.OK

        # Save parameters
        self.MSfile = MSfile
        self.RBfile = RBfile
        self.maxtime = maxtime
        self.MStimestep = MStimestep
        self.RBtimestep = RBtimestep
        self.runRB = runRB
        self.bhcheck = bhcheck
        self.jumptime = jumptime

    def construct_init_data(self, deltam0, grid):
        """
        Constructs initial data based on deltam0 values on a grid
        Assumes that the first grid point is at A > 0
        """

        # Initialize a differentiator
        diff = Derivative(4)
        diff.set_x(grid, 1)

        # Compute dm
        dm = diff.dydx(deltam0)

        # Initial data
        deltam1 = 1.0 * deltam0
        deltau1 = - 0.25 * deltam0
        deltarho1 = deltam0 + grid * dm / 3
        deltar1 = -1/8*(deltam0 + deltarho1)

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

        # Construct the data object
        self.data = ms.Data()
        self.data.umr = np.concatenate((u, m, r))
        self.data.bhcheck = self.bhcheck
        self.data.initialize(0.0)

    def run(self):
        """Runs MS evolution and then RB evolution if applicable"""
        # Start by performing the MS evolution
        print("Beginning MS evolution")
        self._run_MS()

        # Check to see what our status is
        if self.status == Status.MS_IntegrationError:
            print("Unable to integrate further")
            return
        elif self.status == Status.MS_MaxTimeReached:
            print("Maximum time reached")
            return
        elif self.status == Status.OK:
            print("Not sure how we got here...")
            return
        elif self.status == Status.BlackHoleFormed:
            print("Black hole detected!")
            if not self.runRB:
                return
            print("Beginning RB evolution")
            self._init_RB()
            self._run_RB()
            if self.status == Status.RB_IntegrationError:
                print("Unable to integrate further")
                return
            elif self.status == Status.RB_MaxTimeReached:
                print("Maximum time reached")
                return
            elif self.status == Status.MassExtracted:
                print("Mass computed!")
                return

    def _run_MS(self):
        """Perform the Misner-Sharp evolution"""
        if self.data.integrator.t >= self.jumptime:
            # Write the initial conditions
            self.data.write_data(self.MSfile)

        while self.data.integrator.t < self.maxtime :
            # Construct the time to step to
            newtime = self.data.integrator.t + self.MStimestep
            if newtime > self.maxtime :
                newtime = self.maxtime

            # Take a step
            try:
                self.data.step(newtime)
            except IntegrationError:
                self.status = Status.MS_IntegrationError
                return
            except BlackHoleFormed:
                self.status = Status.BlackHoleFormed
            except ShellCrossing:
                self.status = Status.MS_ShellCrossing

            # Write the data
            if self.data.integrator.t > self.jumptime:
                self.data.write_data(self.MSfile)
            print("MS Time:", round(self.data.integrator.t, 5))

            # Get out if status is not OK
            if self.status != Status.OK:
                return

        # If we got here, we ran out of time
        self.status = Status.MS_MaxTimeReached

    def _init_RB(self):
        """Initialize the Russel-Bloomfield evolution"""
        # Initialize the RB data object
        self.rbdata = rb.Data()
        self.rbdata.umrrho = np.concatenate((self.data.u, self.data.m, self.data.r, self.data.rho))
        self.rbdata.initialize(self.data.integrator.t)

        # Set up the RB transition parameters
        self.rbdata.transitionR = self.data.r[np.where(self.data.csp < 0)[0][-1]]
        print("Transition radius:", self.rbdata.transitionR)
        self.rbdata.xi0 = self.data.integrator.t

    def _run_RB(self):
        """Perform the Russel-Bloomfield evolution"""
        # Write the initial conditions if not writing to the MS file
        if self.rbdata.integrator.t >= self.jumptime:
            if self.MSfile is not self.RBfile:
                self.rbdata.write_data(self.RBfile)

        while self.rbdata.integrator.t < self.maxtime :
            # Construct the time to step to
            newtime = self.rbdata.integrator.t + self.RBtimestep
            if newtime > self.maxtime :
                newtime = self.maxtime

            # Take a step
            try:
                self.rbdata.step(newtime)
            except IntegrationError:
                self.status = Status.RB_IntegrationError
                return
            except ShellCrossing:
                self.status = Status.RB_ShellCrossing

            # Write the data
            if self.rbdata.integrator.t >= self.jumptime:
                self.rbdata.write_data(self.RBfile)
            print("RB Time:", round(self.rbdata.integrator.t, 5))

            # Get out if status is not BlackHoleFormed
            if self.status != Status.BlackHoleFormed:
                return

        # If we got here, we ran out of time
        self.status = Status.RB_MaxTimeReached
        return

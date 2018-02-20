#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper that drives MS and RB evolution
"""
from enum import Enum
from ms import MSData, IntegrationError, BlackHoleFormed, NegativeDensity
from rb import RBData


class Status(Enum):
    """Status of Driver object"""
    MS_OK = 0
    BlackHoleFormed = 1
    RB_OK = 2
    MassExtracted = 3
    MS_MaxTimeReached = 4
    RB_MaxTimeReached = 5
    MS_IntegrationError = -1
    RB_IntegrationError = -2
    MS_NegativeDensity = -3
    RB_NegativeDensity = -4


class Driver(object):
    """
    Sets up and runs the entire black hole evolution
    """

    def __init__(self, r, u, m, xi0=0.0,
                 maxtime=7, jumptime=0,
                 viscosity=2.0, debug=False):
        """
        Set parameters for driving this run

        r - the list of points at specific values of R (increasing)
        u - list of \tilde{U} values at each Rgrid value
        m - list of \tilde{m} values at each Rgrid value

        maxtime - Final time to run to
        jumptime - Time before which no writes take place
        debug - An internal debug flag
        """
        # Save parameters
        self.maxtime = maxtime
        self.jumptime = jumptime
        self.debug = debug

        # Initialize the data objects
        self.MSdata = MSData(r, u, m, debug=debug, xi0=xi0, viscosity=viscosity)
        self.RBdata = None
        self.status = Status.MS_OK
        self.msg = ""
        self.hit50 = False
        self.stalled = False

    def runMS(self, outfile, bhcheck=True, timestep=0.1):
        """
        Runs the Misner-Sharp evolution, outputting the generated data to outfile.

        bhcheck controls whether or not black hole formation is checked during the
        evolution. If false, the evolution will continue until an error is raised.

        timestep is the time step for outputting data.

        The resulting status is stored in self.status.
        """
        # Make sure we're ready to roll
        if self.status != Status.MS_OK:
            raise ValueError("Cannot begin MS evolution as status is not OK.")

        # Write initial data to outfile if applicable
        if self.MSdata.integrator.t >= self.jumptime:
            # Write the initial conditions
            self.MSdata.write_data(outfile)

        # Integration loop
        while self.MSdata.integrator.t < self.maxtime:
            # Construct the time to integrate to
            newtime = min(self.MSdata.integrator.t + timestep, self.maxtime)

            # Take a step
            try:
                self.MSdata.step(newtime, bhcheck)
                # Record how things are going
                self.hit50 = self.MSdata.hit50
                self.stalled = self.MSdata.stalled
            except BlackHoleFormed:
                # Don't exit at this stage; we want to record the data
                self.status = Status.BlackHoleFormed
            except IntegrationError as e:
                self.status = Status.MS_IntegrationError
                self.msg = e.args[0]
                return
            except NegativeDensity:
                self.status = Status.MS_NegativeDensity
                return

            # Write the data
            if self.MSdata.integrator.t >= self.jumptime:
                self.MSdata.write_data(outfile)

            # Get out if a black hole has formed
            if self.status == Status.BlackHoleFormed:
                return

        # If we got here, we've run out of time
        self.status = Status.MS_MaxTimeReached

    def runRB(self, outfile, timestep=0.01, write_initial_data=False):
        """
        Runs the Russel-Bloomfield evolution, outputting the generated data to outfile.

        timestep is the time step for outputting data.

        write_initial_data is used to specify if the inital data should be written
        to the file. This is useful when RB data and MS data are being written to
        separate files.

        The resulting status is stored in self.status.
        """
        # Check that the status is correct
        if self.status != Status.BlackHoleFormed:
            raise ValueError("Cannot begin RB evolution before a black hole forms.")

        # Initialize the RB data object
        self.RBdata = RBData(self.MSdata, self.debug)
        self.status = Status.RB_OK

        # Write the initial conditions if not writing to the MS file
        if self.RBdata.integrator.t >= self.jumptime:
            if write_initial_data:
                self.RBdata.write_data(outfile)

        while self.RBdata.integrator.t < self.maxtime:
            # Construct the time to step to
            newtime = min(self.RBdata.integrator.t + timestep, self.maxtime)

            # Take a step
            try:
                self.RBdata.step(newtime)
            except IntegrationError:
                self.status = Status.RB_IntegrationError
                return
            except NegativeDensity:
                self.status = Status.RB_NegativeDensity
                return

            # Write the data
            if self.RBdata.integrator.t >= self.jumptime:
                self.RBdata.write_data(outfile)

            # Status report
            if self.debug:
                print("RB tau:", round(self.RBdata.integrator.t, 5))

        # If we got here, we've run out of time
        self.status = Status.RB_MaxTimeReached

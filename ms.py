#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Misner-Sharp Evolution
"""

from fancyderivs import Derivative
from dopri5 import DOPRI5, DopriIntegrationError
from math import exp, pi, sqrt
import numpy as np
import random

class IntegrationError(Exception):
    pass

class BlackHoleFormed(Exception):
    pass

class ShellCrossing(Exception):
    pass

class NegativeDensity(Exception):
    pass

class Data(object):
    """Object to store all of the appropriate data"""

    def get_info(self):
        """
        Computes everything about the present state:
        xi
        u, m, r, rho (tilded)
        U, M, R, Rho (full)
        horizon (condition)
        cs0, csp, csm (speed of characteristics)
        """
        # Compute all the variables we need
        xi = self.integrator.t
        u, m, r, rho, ephi, gamma2 = compute_data(xi, self.um, self)
        gamma = np.sqrt(gamma2)

        # Save variables to the class
        self.u = u
        self.r = r
        self.m = m
        self.rho = rho
        self.xi = xi

        # Background evolution
        a = exp(xi / 2)
        H = exp(-xi)
        rhob = 3 * H * H / 8 / pi

        # Now compute the non-tilded versions
        self.rfull = a * r
        self.rhofull = rho * rhob
        self.ufull = self.rfull * H * u
        self.mfull = 4 * pi / 3 * rhob * np.power(self.rfull, 3) * m

        # Horizon condition
        self.horizon = r * r * m * H

        # Three speeds of characteristics (note - these are in (xi, R))
        adot = 0.5 / sqrt(3) * ephi * gamma
        self.cs0 = r*(u*ephi-1)/2  # rdot
        self.csp = self.cs0 + adot
        self.csm = self.cs0 - adot

    def initialize(self, xi0):
        """Initialize integrator and derivatives"""
        # Set up the integrator
        self.integrator = DOPRI5(t0=xi0, init_values=self.um, derivs=derivs,
                                 rtol=1e-8, atol=1e-8, params=self)

        # Set up the differentiator
        self.diff = Derivative(4)
        self.newdiff = Derivative(6)
        # Set up for derivatives with respect to r
        self.diff.set_x(self.r, 1)
        self.newdiff.set_x(self.r, 1)

        # Set up for CFL checks
        firstpoint = np.array([self.r[0] * 2])
        self.rdiff = np.concatenate((firstpoint, np.diff(self.r)))

    def step(self, newtime):
        """Takes a step forwards in time"""
        count = 0
        while self.integrator.t < newtime:
            count += 1
            try:
                self.integrator.step()
            except DopriIntegrationError as err:
                raise IntegrationError(err.args[0])

            # Store result
            self.um = self.integrator.values

            # Check if a black hole is present
            if self.blackholecheck() == -1:
                raise BlackHoleFormed

            # Change CFL condition
            self.integrator.update_max_h(self.cfl_check())

        print(count, self.integrator.hdid)

    def cfl_check(self):
        """Check the CFL condition and return the max step size allowed"""
        xi = self.integrator.t
        u, m, r, rho, ephi, gamma2 = compute_data(xi, self.um, self)
        gamma = np.sqrt(gamma2)

        adot = 0.5 / sqrt(3) * ephi * gamma
        cs0 = r*(u*ephi-1)/2  # rdot

        # These are the speeds
        csp = cs0 + adot
        csm = cs0 - adot

        s1 = np.min(self.rdiff / np.abs(csp))
        s2 = np.min(self.rdiff / np.abs(csm))

        return min(s1, s2) * 0.05

    def blackholecheck(self):
        """Returns -1 if an apparent horizon is detected, 0 otherwise"""
        if not self.bhcheck:
            return 0

        # Grab u, m
        u, m = get_um(self.um)

        # Horizon condition
        horizon = self.r*self.r*m*exp(-self.xi)

        # Go and check everything
        for i, val in enumerate(horizon):
            if val >= 1 and u[i] < 0:
                # Apparent horizon detected
                return -1

        # All clear
        return 0

    def write_data(self, file):
        """Writes data to an open file handle"""
        self.get_info()

        file.write("A\tr\tu\tm\trho\tR\tU\tM\tRho\tHorizon\tcsp\tcsm\tcs0\txi\n")
        for i in range(len(self.r)):
            dat = [i,  # 1
                   self.r[i],  # 2
                   self.u[i],  # 3
                   self.m[i],  # 4
                   self.rho[i],  # 5
                   self.rfull[i],  # 6
                   self.ufull[i],  # 7
                   self.mfull[i],  # 8
                   self.rhofull[i],  # 9
                   self.horizon[i],  # 10
                   self.csp[i],  # 11
                   self.csm[i],  # 12
                   self.cs0[i],  # 13
                   self.xi,  # 14
                   ]
            file.write("\t".join(map(str, dat)) + "\n")
        file.write("\n")

def get_um(um):
    """Separates u, m from the composite um object"""
    gridpoints = int(len(um) / 2)
    u = um[0:gridpoints]
    m = um[gridpoints:2*gridpoints]
#    r = umr[2*gridpoints:]
    return u, m

def compute_data(xi, um, data):
    """
    Computes u, m, r, rho, ephi and gamma2
    Initializes derivatives with respect to r in data.diff
    """
    # Start by separating u, m
    u, m = get_um(um)

    # Get R
    r = data.r

    # Check monotonicity of r (shell crossing)
    if np.any(np.diff(r) < 0):
        raise ShellCrossing()

    # Compute dm/dr
    dm = data.diff.dydx(m)

    # Compute various auxiliary variables
    rho = m + r * dm / 3
    if np.any(rho < 0):
        raise NegativeDensity()
    ephi = np.power(rho, -1/4)
    gamma2 = exp(xi) + r*r*(u*u-m)

    # Return the results
    return u, m, r, rho, ephi, gamma2

def derivs(um, xi, data):
    """Computes derivatives for evolution"""
    # Compute all the variables about the present state
    u, m, r, rho, ephi, gamma2 = compute_data(xi, um, data)

    # Compute drho/dr
    drho = data.diff.dydx(rho)
    # drho2 = data.newdiff.dydx(rho)
    # if random.random() < 0.01:
    #     print((drho2[0] - drho[0])/rho[0], (drho2[20] - drho[20])/rho[20])

    # Compute the equations of motion
    (udot, mdot) = compute_eoms(xi, u, m, r, rho, drho, ephi, gamma2, data.diff)
    return np.concatenate((udot, mdot))

def compute_eoms(xi, u, m, r, rho, drho, ephi, gamma2, diff):
    """Computes the equations of motion, given all of the data required to do so"""

    # Compute the time derivatives
    mdot = 2*m - 1.5*u*ephi*(rho/3 + m)
    rdot = r*(u*ephi-1)/2
    udot = u - 0.5*ephi*(0.5*(2*u*u+m+rho) + gamma2*drho/4/rho/r)

    # EULER
    dmdr = diff.dydx(m)
    dudr = diff.dydx(u)

    mdot = mdot - dmdr * rdot
    udot = udot - dudr * rdot
    rdot = rdot * 0

    # Better boundary condition
    c = exp(0.5 * xi) / np.sqrt(12)
    deltam = m[-1] - 1
    uprime = diff.rightdydx(u)
    mprime = diff.rightdydx(m)

    udot[-1] = -0.25 * deltam + (0.25 - c / (2 * r[-1])) * c * mprime + c * mdot[-1] / (2 * r[-1]) - c * uprime
    udot[-1] = 0

    return (udot, mdot)

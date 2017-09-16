#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Misner-Sharp Evolution
"""

from fancyderivs import Derivative
from scipy.integrate import ode
from math import exp, pi, sqrt
import numpy as np

oldtime = 0
newtime = 0

class IntegrationError(Exception):
    pass

class BlackHoleFormed(Exception):
    pass

class Data(object) :
    """Object to store all of the appropriate data"""

    def get_info(self) :
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
        u, m, r, rho, ephi, gamma2 = compute_data(xi, self.umr, self)
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
        self.cs0 = r*(u*ephi-1)/2 # rdot
        self.csp = self.cs0 + adot
        self.csm = self.cs0 - adot

    def initialize(self, xi0) :
        """Initialize integrator and derivatives"""
        # Set up the integrator
        self.integrator = ode(derivs).set_integrator('dopri5', nsteps=10000, rtol=1e-8, atol=1e-8)
        self.integrator.set_initial_value(self.umr, xi0).set_f_params(self)
        if self.bhcheck :
            # Check for a black hole after every internal step
            self.integrator.set_solout(bhcheck)

        # Set up the differentiator
        self.diff = Derivative(4)

    def step(self, newtime) :
        """Takes a step forwards in time"""
        result = self.integrator.integrate(newtime)
        if self.integrator.successful() :
            self.umr = result # Store result
            if self.bhcheck and bhcheck(self.integrator.t, self.umr) == -1 :
                raise BlackHoleFormed
        else :
            raise IntegrationError

    def write_data(self, file) :
        """Writes data to an open file handle"""
        global oldtime, newtime

        self.get_info()

        file.write("# A\tr\tu\tm\trho\tR\tU\tM\tRho\tHorizon\tcsp\tcsm\tcs0\txi\n")
        for i in range(len(self.r)) :
            if i == 0:
                dat = [i, #1
                        self.r[i], #2
                        self.u[i], #3
                        self.m[i], #4
                        self.rho[i], #5
                        self.rfull[i], #6
                        self.ufull[i], #7
                        self.mfull[i], #8
                        self.rhofull[i], #9
                        self.horizon[i], #10
                        self.csp[i], #11
                        self.csm[i], #12
                        self.cs0[i], #13
                        self.xi, #14
                        newtime - oldtime, #15
                        2 * self.r[i] #16
                        ]

            else:
                dat = [i, #1
                        self.r[i], #2
                        self.u[i], #3
                        self.m[i], #4
                        self.rho[i], #5
                        self.rfull[i], #6
                        self.ufull[i], #7
                        self.mfull[i], #8
                        self.rhofull[i], #9
                        self.horizon[i], #10
                        self.csp[i], #11
                        self.csm[i], #12
                        self.cs0[i], #13
                        self.xi, #14
                        newtime - oldtime, #15
                        self.r[i] - self.r[i-1] #16
                        ]
            file.write("\t".join(map(str,dat)) + "\n")
        file.write("\n\n")

def get_umr(umr) :
    """Separates u, m and r from the composite umr object"""
    gridpoints = int(len(umr) / 3)
    u = umr[0:gridpoints]
    m = umr[gridpoints:2*gridpoints]
    r = umr[2*gridpoints:]
    return u, m, r

def compute_data(xi, umr, data) :
    """
    Computes u, m, r, rho, ephi and gamma2
    Initializes derivatives with respect to r in data.diff
    """
    # Start by separating u, m and r
    u, m, r = get_umr(umr)

    # Set up for derivatives with respect to r
    data.diff.set_x(r, 1)

    # Compute dm/dr
    dm = data.diff.dydx(m)

    # Compute various auxiliary variables
    rho = m + r * dm / 3
    # TODO Check for negative rho
    ephi = np.power(rho, -1/4)
    gamma2 = exp(xi) + r*r*(u*u-m)

    # Return the results
    return u, m, r, rho, ephi, gamma2

def derivs(xi, umr, data) :
    """Computes derivatives for evolution"""
    # Compute all the variables about the present state
    u, m, r, rho, ephi, gamma2 = compute_data(xi, umr, data)

    # Compute drho/dr
    # Note that compute_data already set_x for the derivative
    drho = data.diff.dydx(rho)

    # Compute the time derivatives
    mdot = 2*m - 1.5*u*ephi*(rho/3 + m)
    rdot = r*(u*ephi-1)/2
    udot = u - 0.5*ephi*(0.5*(2*u*u+m+rho) + gamma2*drho/4/rho/r)

    # Basic reflecting boundary condition
    #mdot[-1] = 0

    # better boundary condition
    c = exp(0.5 * xi) / np.sqrt(12)
    deltam = m[-1] - 1
    uprime = data.diff.rightdydx(u)
    mprime = data.diff.rightdydx(m)

    udot[-1] = -0.25 * deltam + (0.25 - c / (2 * r[-1])) * c * mprime + c * mdot[-1] / (2 * r[-1]) - c * uprime

    return np.concatenate((udot, mdot, rdot))

def bhcheck(xi, umr) :
    """Returns -1 if an apparent horizon is detected, 0 otherwise"""
    global oldtime, newtime
#    print(oldtime, newtime)

    if xi > newtime:
        oldtime = newtime
        newtime = xi

    # Grab u, m and r
    u, m, r = get_umr(umr)

    # Horizon condition
    horizon = r*r*m*exp(-xi)

    # Go and check everything
    for i, val in enumerate(horizon) :
        if val >= 1 and u[i] < 0 :
            # Apparent horizon detected
            return -1

    # All clear
    return 0

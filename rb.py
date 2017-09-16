#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Russel-Bloomfield Evolution
"""

from fancyderivs import Derivative
from scipy.integrate import ode
from math import pi, tan, tanh, cosh, sin, sqrt
import numpy as np
from ms import IntegrationError

class Data(object) :
    """Object to store all of the appropriate data"""

    def initialize(self, tau0) :
        """Initialize integrator and derivatives"""
        # Set up the integrator
        self.integrator = ode(derivs).set_integrator('dopri5', nsteps=10000, rtol=1e-8, atol=1e-8)
        self.integrator.set_initial_value(self.umrrho, tau0).set_f_params(self)

        # Set up the differentiator
        self.diff = Derivative(4)

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
        tau = self.integrator.t
        u, m, r, rho, ephi, gamma2, xi = compute_data(tau, self.umrrho, self)
        gamma = np.sqrt(gamma2)

        # Save variables to the class
        self.u = u
        self.r = r
        self.m = m
        self.rho = rho
        self.xi = xi

        # Background evolution
        a = np.exp(xi / 2)
        H = np.exp(-xi)
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

    def step(self, newtime) :
        """Takes a step forwards in time"""
        result = self.integrator.integrate(newtime)
        if self.integrator.successful() :
            self.umrrho = result
            # TODO shell crossing check
        else :
            raise IntegrationError

    def write_data(self, file) :
        """Writes data to an open file handle"""
        self.get_info()

        file.write("# A\tr\tu\tm\trho\tR\tU\tM\tRho\tHorizon\tcsp\tcsm\tcs0\txi\n")
        for i in range(len(self.r)) :
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
                   self.xi[i] #14
                  ]
            file.write("\t".join(map(str,dat)) + "\n")
        file.write("\n\n")
        # TODO Also want to output the various denominators that are used in our computations

def get_umrrho(umrrho) :
    """Separates u, m, r and rho from the composite umrrho object"""
    gridpoints = int(len(umrrho) / 4)
    u = umrrho[0:gridpoints]
    m = umrrho[gridpoints:2*gridpoints]
    r = umrrho[2*gridpoints:3*gridpoints]
    rho = umrrho[3*gridpoints:]
    return u, m, r, rho

def compute_data(tau, umrrho, data) :
    """
    Computes u, m, r, rho, ephi, gamma2 and xi
    Initializes derivatives with respect to r in data.diff
    """
    # Start by separating u, m, r and rho
    u, m, r, rho = get_umrrho(umrrho)

    # Compute xi
    xi = np.array([ffunc(tau, ri, data) for ri in r])

    # Set up for derivatives with respect to r
    data.diff.set_x(r, 1)

    # Compute various auxiliary variables
    ephi = np.power(rho, -1/4)
    gamma2 = np.exp(xi) + r*r*(u*u-m)

    # Return the results
    return u, m, r, rho, ephi, gamma2, xi

def derivs(tau, umrrho, data) :
    """Computes derivatives for evolution"""
    # Compute all the information about the present state
    u, m, r, rho, ephi, gamma2, xi = compute_data(tau, umrrho, data)

    # Compute derivatives of f
    dfdr = np.array([fprimefunc(tau, ri, data) for ri in r])
    partialdfdtau = np.array([fdotfunc(tau, ri, data) for ri in r])
    temp = r * (u * ephi - 1) / 2
    chunk = 1 - dfdr * temp
    dfdtau = partialdfdtau / chunk

    # Compute derivatives
    du = data.diff.dydx(u)
    drho = data.diff.dydx(rho)

    # Compute the pieces of the equations of motion for u and rho
    A = dfdtau * (u - 0.25 * ephi * (2*u*u+m+rho))
    B = -dfdtau * ephi * gamma2 / chunk / r / rho / 8
    C = 2*dfdtau*rho*(1-ephi*u)
    D = - 2 * dfdtau * rho * ephi * r / 3 / chunk
    newchunk = dfdtau * dfdtau - B * D * dfdr * dfdr

    # TODO Write chunk and newchunk in terms of conditions on characteristics
    # Simplify their computation (should be expressable in terms of nice quantities)
    # Put in a function
    # Save appropriate denominators in get_info, and output when printing

    # Check for any fdot values that are zero at low A
    # This will cause newchunk to be vanishing!
    for i, val in enumerate(dfdtau) :
        if val == 0.0 :
            newchunk[i] = 1.0
            # As fdot is zero here, udot and rhodot are zero automatically
            # and modifying newchunk doesn't change anything
        else :
            # No more zeros to worry about; dfdtau may only vanish near the origin
            break

    # Compute the equations of motion for everything
    rdot = dfdtau*temp
    mdot = dfdtau*(2*m - 1.5*u*ephi*(rho/3 + m))
    udot = dfdtau / newchunk * (A*dfdtau - B*C*dfdr - B*D*du*dfdr + B*dfdtau*drho)
    rhodot = dfdtau / newchunk * (C*dfdtau - A*D*dfdr - B*D*drho*dfdr + D*dfdtau*du)

    # Basic reflecting boundary condition
    mdot[-1] = 0
    rhodot[-1] = 0

    return np.concatenate((udot, mdot, rdot, rhodot))

def csc(val) :
    """Returns cosec of the value"""
    return 1.0 / sin(val)

def cot(val) :
    """Returns cot of the value"""
    return 1.0 / tan(val)

def sech(val) :
    """Returns sech of the value"""
    try:
        return 1.0 / cosh(val)
    except OverflowError :
        return 0.0

def ffunc(tau, r, data) :
    """Computes xi = f(tau, r)"""
    #return tau # Uncomment to use MS evolution (and in fprimefunc, fdotfunc)
    if r >= data.A1 :
        return tau
    if r <= data.A0 :
        return data.xi0
    return data.xi0 + 0.5 * (tau - data.xi0) * (tanh(tan(pi*((r - data.A0)/(data.A1 - data.A0) - 0.5))) + 1)

def fprimefunc(tau, r, data) :
    """Computes df/dr"""
    #return 0.0 # Uncomment to use MS evolution
    if r >= data.A1 :
        return 0.0
    if r <= data.A0 :
        return 0.0
    val = pi * (r - data.A0) / (data.A1 - data.A0)
    return 0.5 * (tau - data.xi0) * pi / (data.A1 - data.A0) * csc(val)**2 * sech(cot(val))**2

def fdotfunc(tau, r, data) :
    """Computes df/dtau (partial derivatives)"""
    #return 1.0 # Uncomment to use MS evolution
    if r >= data.A1 :
        return 1.0
    if r <= data.A0 :
        return 0.0
    return 0.5 * (tanh(tan(pi*((r - data.A0)/(data.A1 - data.A0) - 0.5))) + 1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Russel-Bloomfield Evolution
"""

from fancyderivs import Derivative
from scipy.integrate import ode
from math import pi, tan, tanh, cosh, sin, sqrt
import numpy as np
from ms import IntegrationError, ShellCrossing, NegativeDensity, compute_eoms

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

    # Check monotonicity of r (shell crossing)
    if np.any(np.diff(r) < 0):
        raise ShellCrossing()

    # Check for negative rho
    if np.any(rho < 0):
        raise NegativeDensity()

    # Compute xi
    xi = np.array([ffunc(tau, ri, data.transitionR, data.xi0) for ri in r])

    # Compute various auxiliary variables
    ephi = np.power(rho, -1/4)
    gamma2 = np.exp(xi) + r*r*(u*u-m)

    # Return the results
    return u, m, r, rho, ephi, gamma2, xi

def derivs(tau, umrrho, data) :
    """Computes derivatives for evolution"""
    # Compute all the information about the present state
    u, m, r, rho, ephi, gamma2, xi = compute_data(tau, umrrho, data)

    # Find the index at which the split into RB and MS occurs
    transindex = np.where(r <= data.transitionR)[0][-1] + 1

    # Extract the MS values
    MSu = u[transindex:]
    MSm = m[transindex:]
    MSr = r[transindex:]
    MSrho = rho[transindex:]
    MSephi = ephi[transindex:]
    MSgamma2 = gamma2[transindex:]
    # Compute MS drho and du, using no boundary at the leftmost index
    data.diff.set_x(MSr, 0)
    MSdrho = data.diff.dydx(MSrho)
    MSdu = data.diff.dydx(MSu)
    # Compute the MS EOMs
    MSudot, MSmdot, MSrdot = compute_eoms(tau, MSu, MSm, MSr, MSrho, MSdrho, MSephi, MSgamma2, data.diff)
    # Also need to compute MSrhodot
    MSrhodot =2*MSrho*(1 - MSephi * (MSu + MSr*MSdu/3))

    # Extract the RB values
    RBm = m[0:transindex]
    RBu = u[0:transindex + 2]
    RBr = r[0:transindex + 2]
    RBrho = rho[0:transindex + 2]
    RBephi = ephi[0:transindex]
    RBgamma2 = gamma2[0:transindex]
    # Compute RB derivatives (carefully - using two extra gridpoints for best derivatives)
    data.diff.set_x(RBr, 1)
    RBdu = data.diff.dydx(RBu)
    RBdrho = data.diff.dydx(RBrho)
    # Now lop off the last two datapoints that are now unnecessary
    RBdu = RBdu[0:-2]
    RBdrho = RBdrho[0:-2]
    RBu = RBu[0:-2]
    RBr = RBr[0:-2]
    RBrho = RBrho[0:-2]

    # Now compute the RB equations of motion
    # Compute derivatives of f
    dfdr = np.array([fprimefunc(tau, ri, data.transitionR, data.xi0) for ri in RBr])
    partialdfdtau = np.array([fdotfunc(tau, ri, data.transitionR, data.xi0) for ri in RBr])
    temp = RBr * (RBu * RBephi - 1) / 2
    chunk = 1 - dfdr * temp
    dfdtau = partialdfdtau / chunk

    # Compute the pieces of the equations of motion for u and rho
    A = dfdtau * (RBu - 0.25 * RBephi * (2*RBu*RBu+RBm+RBrho))
    B = -dfdtau * RBephi * RBgamma2 / chunk / RBr / RBrho / 8
    C = 2*dfdtau*RBrho*(1-RBephi*RBu)
    D = - 2 * dfdtau * RBrho * RBephi * RBr / 3 / chunk
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
    RBrdot = dfdtau*temp
    RBmdot = dfdtau*(2*RBm - 1.5*RBu*RBephi*(RBrho/3 + RBm))
    RBudot = dfdtau / newchunk * (A*dfdtau - B*C*dfdr - B*D*RBdu*dfdr + B*dfdtau*RBdrho)
    RBrhodot = dfdtau / newchunk * (C*dfdtau - A*D*dfdr - B*D*RBdrho*dfdr + D*dfdtau*RBdu)

    # Combine the results to give the full vector of derivatives
    return np.concatenate((RBudot, MSudot, RBmdot, MSmdot, RBrdot, MSrdot, RBrhodot, MSrhodot))

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

def ffunc(tau, r, transitionR, xi0) :
    """Computes xi = f(tau, r)"""
    if r >= transitionR :
        return tau
    return xi0 + 0.5 * (tau - xi0) * (tanh(tan(pi*(r / transitionR - 0.5))) + 1)

def fprimefunc(tau, r, transitionR, xi0) :
    """Computes df/dr inside the transition"""
    val = pi * r / transitionR
    return 0.5 * (tau - xi0) * pi / transitionR * csc(val)**2 * sech(cot(val))**2

def fdotfunc(tau, r, transitionR, xi0) :
    """Computes df/dtau (partial derivatives) inside the transition"""
    return 0.5 * (tanh(tan(pi*(r / transitionR - 0.5))) + 1)

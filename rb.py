#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Russel-Bloomfield Evolution
"""

from math import pi, tan, tanh, cosh, sin, sqrt, exp
import numpy as np
from ms import IntegrationError, NegativeDensity
from fancyderivs import Derivative
from dopri5 import DOPRI5, DopriIntegrationError


class RBData(object):
    """Object to store all of the appropriate data"""

    def __init__(self, MSdata, debug=False):
        """Initialize integrator and derivatives"""
        self.debug = debug

        # Grab the initial data
        self.xi0 = MSdata.integrator.t
        self.r = MSdata.r
        self.umrho = np.concatenate((MSdata.u, MSdata.m, MSdata.rho))

        # Set up the RB transition parameters
        self.transitionR = self.r[np.where(MSdata.csp < 0)[0][-1]]

        # Set up the integrator
        self.integrator = DOPRI5(t0=self.xi0, init_values=self.umrho, derivs=derivs,
                                 rtol=1e-8, atol=1e-8, params=self)

        # Set up for derivatives with respect to r
        self.diff = Derivative(4)
        self.diff.set_x(self.r, 1)

        # Set up for CFL checks
        firstpoint = np.array([self.r[0] * 2])
        self.rdiff = np.concatenate((firstpoint, np.diff(self.r)))

        if self.debug:
            print("Transition radius:", self.transitionR)

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
        tau = self.integrator.t
        u, m, r, rho, ephi, gamma2, xi = compute_data(tau, self.umrho, self)
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
        self.cs0 = r*(u*ephi-1)/2  # rdot
        self.csp = self.cs0 + adot
        self.csm = self.cs0 - adot

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
            self.umrho = self.integrator.values

            # Change CFL condition
            self.integrator.update_max_h(self.cfl_check())

        print(count, self.integrator.hdid)

    def cfl_check(self):
        """Check the CFL condition and return the max step size allowed"""
        tau = self.integrator.t
        u, m, r, rho, ephi, gamma2, xi = compute_data(tau, self.umrho, self)
        if np.any(gamma2 < 0):
            raise IntegrationError("Gamma^2 went negative")
        gamma = np.sqrt(gamma2)

        adot = 0.5 / sqrt(3) * ephi * gamma
        cs0 = r*(u*ephi-1)/2  # rdot

        # These are the speeds
        csp = cs0 + adot
        csm = cs0 - adot

        s1 = np.min(self.rdiff / np.abs(csp))
        s2 = np.min(self.rdiff / np.abs(csm))
        fdot = np.ones_like(self.r)
        for idx, r in enumerate(self.r):
            if r >= self.transitionR:
                break
            fdot[idx] = fdotfunc(tau, r, self.transitionR, self.xi0)

        return min(s1, s2) * 0.05

    def write_data(self, file):
        """Writes data to an open file handle"""
        self.get_info()

        file.write("A\tr\tu\tm\trho\tR\tU\tM\tRho\tHorizon\tcsp\tcsm\tcs0\txi\n")
        for i in range(len(self.r)):
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
        file.write("\n")

def get_umrho(umrho):
    """Separates u, m, r and rho from the composite umrrho object"""
    gridpoints = int(len(umrho) / 3)
    u = umrho[0:gridpoints]
    m = umrho[gridpoints:2*gridpoints]
    rho = umrho[2*gridpoints:3*gridpoints]
    return u, m, rho

def compute_data(tau, umrho, data):
    """
    Computes u, m, r, rho, ephi, gamma2 and xi
    Initializes derivatives with respect to r in data.diff
    """
    # Start by separating u, m, r and rho
    u, m, rho = get_umrho(umrho)
    r = data.r

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

def derivs(umrho, tau, data):
    """Computes derivatives for evolution"""
    # Compute all the information about the present state
    u, m, r, rho, ephi, gamma2, xi = compute_data(tau, umrho, data)

    # Find the index at which the split into RB and MS occurs
    transindex = np.where(r <= xb(xi, data.xi0, data.transitionR))[0][-1] + 1

    drho = data.diff.dydx(rho)
    du = data.diff.dydx(u)
    dm = data.diff.dydx(m)

    # Extract the MS values
    MSu = u[transindex:]
    MSm = m[transindex:]
    MSr = r[transindex:]
    MSrho = rho[transindex:]
    MSephi = ephi[transindex:]
    MSgamma2 = gamma2[transindex:]
    MSdrho = drho[transindex:]
    MSdu = du[transindex:]
    MSdm = dm[transindex:]
    # Compute the MS EOMs
    MSudot, MSmdot = compute_eoms(tau, MSu, MSm, MSr, MSrho, MSdrho, MSephi,
                                  MSgamma2, data.diff, MSdu, MSdm)
    # Also need to compute MSrhodot
    MSrhodot = 2*MSrho*(1 - MSephi * (MSu + MSr*MSdu/3))

    # Extract the RB values
    RBm = m[0:transindex]
    RBu = u[0:transindex]
    RBr = r[0:transindex]
    RBrho = rho[0:transindex]
    RBephi = ephi[0:transindex]
    RBgamma2 = gamma2[0:transindex]
    RBdu = du[0:transindex]
    RBdrho = drho[0:transindex]

    # Now compute the RB equations of motion
    # Compute derivatives of f
    dfdr = np.array([fprimefunc(tau, ri, data.transitionR, data.xi0) for ri in RBr])
    partialdfdtau = np.array([fdotfunc(tau, ri, data.transitionR, data.xi0) for ri in RBr])
    temp = RBr * (RBu * RBephi - 1) / 2
    chunk = 1 - dfdr * temp
    dfdtau = partialdfdtau / chunk

    # Compute the pieces of the equations of motion for u and rho
    #A = dfdtau * (RBu - 0.25 * RBephi * (2*RBu*RBu+RBm+RBrho))
    #B = -dfdtau * RBephi * RBgamma2 / chunk / RBr / RBrho / 8
    #C = 2*dfdtau*RBrho*(1-RBephi*RBu)
    #D = - 2 * dfdtau * RBrho * RBephi * RBr / 3 / chunk
    #newchunk = dfdtau * dfdtau - B * D * dfdr * dfdr

    rspeed = 0.5 * RBr * (RBu * RBephi - 1)

    A = RBu - 0.5*RBephi * ( RBgamma2 * RBdrho/ (4 * RBr * RBrho) + 0.5 * (2 * RBu**2 +RBm + RBrho)) - rspeed * RBdu
    B = dfdr * rspeed
    C = 3 * 0.25 * RBephi * RBgamma2 * dfdr / 3 / (2 * RBr * RBrho)
    D = 2 * RBrho * (1 - RBephi * (RBu + RBr * RBdu / 3)) - rspeed * RBdrho
    E = 2 * RBrho * RBephi * dfdr * RBr / 3

    wavespeed2 = 0.25 * RBephi * RBephi * RBgamma2 / 3
    denominator = ((1 - B)**2 - dfdr**2 * wavespeed2)


    # TODO Write chunk and newchunk in terms of conditions on characteristics
    # Simplify their computation (should be expressable in terms of nice quantities)
    # Put in a function
    # Save appropriate denominators in get_info, and output when printing

    # Check for any fdot values that are zero at low A
    # This will cause newchunk to be vanishing!
    for i, val in enumerate(dfdtau):
        if val == 0.0 :
            denominator = 1.0
            # As fdot is zero here, udot and rhodot are zero automatically
            # and modifying newchunk doesn't change anything
        else :
            # No more zeros to worry about; dfdtau may only vanish near the origin
            break

    # Compute the equations of motion for everything
#    RBrdot = dfdtau*temp
    RBmdot = dfdtau*(2*RBm + 1.5*(RBrho - RBm) - 2*RBu*RBephi*RBrho)
    RBudot = dfdtau / denominator * (A * (1-B) + C*D)
    RBrhodot = dfdtau / denominator * (D * (1-B) + A*E)

    # Combine the results to give the full vector of derivatives
    return np.concatenate((RBudot, MSudot, RBmdot, MSmdot, RBrhodot, MSrhodot))

def csc(val):
    """Returns cosec of the value"""
    return 1.0 / sin(val)

def cot(val):
    """Returns cot of the value"""
    return 1.0 / tan(val)

def sech(val):
    """Returns sech of the value"""
    try:
        return 1.0 / cosh(val)
    except OverflowError :
        return 0.0

def xb(tau, xi0, x0):
    return x0 * np.exp(0.5 * (xi0 - tau))

def ffunc(tau, r, x0, xi0):
    """Computes xi = f(tau, r)"""
    if r >= xb(tau,xi0,x0):
        return tau
    return xi0 + 0.5 * (tau - xi0) * (tanh(tan(pi*(r / xb(tau,xi0,x0) - 0.5))) + 1)

def fprimefunc(tau, r, x0, xi0):
    """Computes df/dr inside the transition"""
    val = r / xb(tau, xi0, x0)
    if val >= 1:
        return 0
    val = pi * val
    return 0.5 * (tau - xi0) * pi / xb(tau,xi0,x0) * csc(val)**2 * sech(cot(val))**2

def fdotfunc(tau, r, x0, xi0):
    """Computes df/dtau (partial derivatives) inside the transition"""
    val = r / xb(tau, xi0, x0)
    if val >= 1:
        return 1
    val = pi * val
    return 0.5*(1 - tanh(cot(val))) + (tau - xi0) * 0.5 * 0.5 * val * sech(cot(val))**2 * csc(val)**2

def compute_eoms(xi, u, m, r, rho, drho, ephi, gamma2, diff, dudr, dmdr):
    """Computes the equations of motion, given all of the data required to do so"""

    # Compute the time derivatives
    mdot = 2*m - 1.5*u*ephi*(rho/3 + m)
    rdot = r*(u*ephi-1)/2
    udot = u - 0.5*ephi*(0.5*(2*u*u+m+rho) + gamma2*drho/4/rho/r)

    # EULER
    mdot = mdot - dmdr * rdot
    udot = udot - dudr * rdot
    rdot = rdot * 0

    # Better boundary condition
    c = exp(0.5 * xi) / np.sqrt(12)
    deltam = m[-1] - 1
    uprime = dudr[-1]
    mprime = dmdr[-1]

    udot[-1] = -0.25 * deltam + (0.25 - c / (2 * r[-1])) * c * mprime + c * mdot[-1] / (2 * r[-1]) - c * uprime

    return (udot, mdot)

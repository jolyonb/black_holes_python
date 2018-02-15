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
    except OverflowError:
        return 0.0

class RBData(object):
    """Object to store all of the appropriate data"""

    def __init__(self, MSdata, debug=False):
        """Initialize integrator and derivatives"""
        self.debug = debug

        # Grab the initial data
        self.xi0 = MSdata.integrator.t
        self.r = MSdata.r
        MSdata.get_info()
        u = MSdata.computed_data["u"]
        m = MSdata.computed_data["m"]
        rho = MSdata.computed_data["rho"]
        self.umrho = np.concatenate((u, m, rho))
        self.gridpoints = len(self.r)

        # Set up the computed data
        self.computed_data = {"tau": -1, "index": np.array([i for i in range(self.gridpoints)])}

        # Set up the RB transition parameters
        csp = MSdata.computed_data["csp"]
        self.transitionR = self.r[np.where(csp < 0)[0][-1]]

        # Set up the integrator
        self.integrator = DOPRI5(t0=self.xi0, init_values=self.umrho, derivs=self.derivs,
                                 rtol=1e-8, atol=1e-8, params=None)

        # Set up for derivatives with respect to r
        self.diff = Derivative(4)
        self.diff.set_x(self.r, 1)

        # Set up for CFL checks
        self.rdiff = np.concatenate((np.array([self.r[0] * 2]), np.diff(self.r)))

        if self.debug:
            print("Transition radius:", self.transitionR)
            print("Transition time:", self.xi0)

    def get_info(self):
        # Check to see if everything has already been computed
        tau = self.integrator.t
        if self.computed_data["tau"] == tau:
            return

        # Compute all the variables we need
        u, m, r, rho, ephi, gamma2, xi, cs0 = self.compute_data(tau, self.umrho)

        if np.any(gamma2 < 0):
            raise IntegrationError("Gamma^2 went negative")

        self.computed_data["tau"] = tau
        self.computed_data["u"] = u
        self.computed_data["m"] = m
        self.computed_data["r"] = r
        self.computed_data["rho"] = rho
        self.computed_data["gamma"] = gamma = np.sqrt(gamma2)
        self.computed_data["ephi"] = ephi
        self.computed_data["xi"] = xi

        # Background evolution
        self.computed_data["a"] = a = np.exp(xi / 2)
        self.computed_data["H"] = H = np.exp(-xi)
        self.computed_data["rhob"] = rhob = 3 * H * H / 8 / pi

        # Now compute the non-tilded versions
        self.computed_data["rfull"] = rfull = a * r
        self.computed_data["rhofull"] = rho * rhob
        self.computed_data["ufull"] = rfull * H * u
        self.computed_data["mfull"] = 4 * pi / 3 * rhob * np.power(rfull, 3) * m

        # Horizon condition
        self.computed_data["horizon"] = r * r * m * H

        # Three speeds of characteristics (note - these are in (xi, R))
        adot = 0.5 / sqrt(3) * ephi * gamma
        self.computed_data["cs0"] = cs0  # rdot
        self.computed_data["csp"] = cs0 + adot
        self.computed_data["csm"] = cs0 - adot

    def step(self, newtime, safety=0.75):
        """Takes a step forwards in time"""
        stepcount = 0
        while self.integrator.t < newtime:
            stepcount += 1
            try:
                self.integrator.step()
            except DopriIntegrationError as err:
                raise IntegrationError(err.args[0])

            # Store result
            self.um = self.integrator.values

            # Change CFL condition
            self.integrator.update_max_h(self.cfl_check(safety))

        if self.debug:
            msg = "RB: Stepped to tau = {} in {} steps, last stepsize was {}"
            print(msg.format(round(self.integrator.t, 5),
                             stepcount,
                             round(self.integrator.hdid, 5)))

    def cfl_check(self, safety):
        """Check the CFL condition and return the max step size allowed"""
        # Get the propagation speeds
        self.get_info()
        csp = np.abs(self.computed_data["csp"])
        csm = np.abs(self.computed_data["csm"])
        tau = self.computed_data["tau"]

        s1 = np.min(self.rdiff / np.abs(csp))
        s2 = np.min(self.rdiff / np.abs(csm))
        fdot = np.ones_like(self.r)
        for idx, r in enumerate(self.r):
            if r >= self.transitionR:
                break
            fdot[idx] = self.fdotfunc(tau, r)

        return min(s1, s2) * safety

    def write_data(self, file):
        """Writes data to an open file handle"""
        # Extract all the data we want
        self.get_info()
        datanames = [   # gnuplot index
            "index",    # 1
            "r",        # 2
            "u",        # 3
            "m",        # 4
            "rho",      # 5
            "rfull",    # 6
            "ufull",    # 7
            "mfull",    # 8
            "rhofull",  # 9
            "horizon",  # 10
            "csp",      # 11
            "csm",      # 12
            "cs0",      # 13
            "xi"        # 14
        ]
        fulldata = [self.computed_data[name] for name in datanames]
        file.write("# " + "\t".join(map(str, datanames)) + "\n")

        # Go and write the block of data
        for i in range(self.gridpoints):
            dat = [data[i] if isinstance(data, np.ndarray) else data for data in fulldata]
            file.write("\t".join(map(str, dat)) + "\n")
        file.write("\n")

    def compute_data(self, tau, umrho):
        """
        Computes u, m, r, rho, ephi, gamma2, xi and cs0
        """
        # Start by getting r, u, m and rho
        r = self.r
        u = umrho[0:self.gridpoints]
        m = umrho[self.gridpoints:2*self.gridpoints]
        rho = umrho[2*self.gridpoints:]

        # Check for negative rho
        if np.any(rho < 0):
            raise NegativeDensity()

        # Compute xi
        xi = np.array([self.ffunc(tau, ri) for ri in r])

        # Compute various auxiliary variables
        ephi = np.power(rho, -1/4)
        gamma2 = np.exp(xi) + r*r*(u*u-m)

        # Compute rdot, which is needed for the characteristic speeds
        rdot = r*(u*ephi-1)/2

        # Return the results
        return u, m, r, rho, ephi, gamma2, xi, rdot

    def boundary_r(self, tau):
        """
        Returns the position of the boundary as a function of tau, located at the
        initial apparent horizon, but shrinking inwards due to cosmic expansion
        """
        return self.transitionR * np.exp(0.5 * (self.xi0 - tau))

    def ffunc(self, tau, r, x0, xi0):
        """
        Computes xi = f(tau, r)

        The transition position r_transition is computed by the boundary function.
        For r > r_transition, ffunc = tau
        For r < r_transition, ffun is a smooth function that interpolates between xi0 and tau
        """
        r_transition = self.boundary_r(tau)
        if r >= r_transition:
            return tau
        return xi0 + 0.5 * (tau - xi0) * (tanh(tan(pi*(r / r_transition - 0.5))) + 1)

    def fprimefunc(self, tau, r):
        """Computes df/dr (partial derivatives) at a given tau and r"""
        r_transition = self.boundary_r(tau)
        if r >= r_transition:
            return 0
        val = pi * r / r_transition
        return 0.5 * (tau - self.xi0) * pi / r_transition * csc(val)**2 * sech(cot(val))**2

    def fdotfunc(self, tau, r):
        """Computes df/dtau (partial derivatives) at a given tau and r"""
        r_transition = self.boundary_r(tau)
        if r >= r_transition:
            return 1
        val = pi * r / r_transition
        return (0.5*(1 - tanh(cot(val))) +
                (tau - self.xi0) * 0.5*0.5 * val * sech(cot(val))**2 * csc(val)**2)

    def derivs(self, umrho, tau, params=None):
        """Computes derivatives for evolution"""
        # Compute all the information about the present state
        u, m, r, rho, ephi, gamma2, xi, cs0 = self.compute_data(tau, umrho)

        # Find the index at which the split into RB and MS occurs
        transindex = np.where(r <= self.boundary_r(xi, self.xi0, data.transitionR))[0][-1] + 1

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
            if val == 0.0:
                denominator = 1.0
                # As fdot is zero here, udot and rhodot are zero automatically
                # and modifying newchunk doesn't change anything
            else:
                # No more zeros to worry about; dfdtau may only vanish near the origin
                break

        # Compute the equations of motion for everything
    #    RBrdot = dfdtau*temp
        RBmdot = dfdtau*(2*RBm + 1.5*(RBrho - RBm) - 2*RBu*RBephi*RBrho)
        RBudot = dfdtau / denominator * (A * (1-B) + C*D)
        RBrhodot = dfdtau / denominator * (D * (1-B) + A*E)

        # Combine the results to give the full vector of derivatives
        return np.concatenate((RBudot, MSudot, RBmdot, MSmdot, RBrhodot, MSrhodot))

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Misner-Sharp Evolution
"""

from math import exp, pi, sqrt, log
import numpy as np
from dopri5 import DOPRI5, DopriIntegrationError
from newderivs import Derivative

class IntegrationError(Exception):
    """An integration error occurred when evolving"""
    pass

class BlackHoleFormed(Exception):
    """A black hole was detected"""
    pass

class NegativeDensity(Exception):
    """A negative energy density was detected"""
    pass

class MSData(object):
    """Object to store all of the appropriate data"""

    def __init__(self, r, u, m, xi0=0.0, viscosity=2.0, debug=False):
        """
        Initialize the data object with the given grid, \tilde{U} and \tilde{m} values
        Also sets up the integrator and differentiator
        """
        self.debug = debug
        self.viscosity = viscosity

        # Store the grid and initial data
        self.r = r
        self.um = np.concatenate((u, m))
        self.gridpoints = len(r)
        self.computed_data = {
            "xi": -1,
            "index": np.array([i for i in range(self.gridpoints)]),
        }

        # Set up the integrator
        self.integrator = DOPRI5(t0=xi0, init_values=self.um, derivs=self.derivs,
                                 rtol=1e-8, atol=1e-8)

        # Set up the differentiator for derivatives with respect to r
        self.diff = Derivative(self.r)

        # Compute the grid spacing (used for CFL checks)
        self.rdiff = np.concatenate((np.array([self.r[0] * 2]), np.diff(self.r)))

        # Did the central density ever hit 50?
        self.hit50 = False
        # Has formation stalled? This is set when central density decreases after hitting 50
        self.stalled = False

    def step(self, newtime, bhcheck=True, safety=0.75):
        """Takes a step forwards in time"""
        stepcount = 0
        lastrho = 0
        while self.integrator.t < newtime:
            stepcount += 1
            try:
                self.integrator.step()
            except DopriIntegrationError as err:
                raise IntegrationError(err.args[0])

            # Store result
            self.um = self.integrator.values

            # Check if a black hole is present
            if bhcheck and self.hasblackhole():
                raise BlackHoleFormed()

            # Change CFL condition
            self.integrator.update_max_h(self.cfl_check(safety))

            # Check for hit50 and stalling
            rhocheck = self.computed_data["rho"][0]
            if rhocheck > 50:
                self.hit50 = True
                if rhocheck < lastrho:
                    self.stalled = True
            lastrho = rhocheck

        if self.debug:
            msg = "MS: Stepped to xi = {} in {} steps, last stepsize was {}"
            print(msg.format(round(self.integrator.t, 5),
                             stepcount,
                             round(self.integrator.hdid, 5)))

    def get_info(self):
        """
        Computes everything about the present state:
        xi
        u, m, r, rho (tilded)
        U, M, R, Rho (full)
        horizon (condition)
        cs0, csp, csm (speed of characteristics)
        """
        # Check to see if everything has already been computed
        xi = self.integrator.t
        if self.computed_data["xi"] == xi:
            return

        # Compute all the variables we need
        u, m, r, rho, ephi, gamma2, _, _, cs0, Q, _, _ = self.compute_data(xi, self.um)
        # gamma2 is \bar{\gamma}^2

        if np.any(gamma2 < 0):
            raise IntegrationError("Gamma^2 went negative")

        self.computed_data["xi"] = xi
        self.computed_data["u"] = u
        self.computed_data["m"] = m
        self.computed_data["r"] = r
        self.computed_data["rho"] = rho
        self.computed_data["gamma"] = gamma = np.sqrt(gamma2)
        self.computed_data["ephi"] = ephi

        # Background evolution
        self.computed_data["a"] = a = exp(xi / 2)
        self.computed_data["H"] = H = 1/(a*a)
        self.computed_data["rhob"] = rhob = 3 * H * H / 8 / pi

        # Now compute the non-tilded versions
        self.computed_data["rfull"] = rfull = a * r
        self.computed_data["rhofull"] = rho * rhob
        self.computed_data["ufull"] = rfull * H * u
        self.computed_data["mfull"] = 4 * pi / 3 * rhob * np.power(rfull, 3) * m

        # Horizon condition
        self.computed_data["horizon"] = r * r * m * H

        # Three speeds of characteristics (note - these are in (xi, R))
        adot = 0.5 / sqrt(3) * ephi * gamma  # Note that this is the linearized speed of sound
        self.computed_data["cs0"] = cs0  # rdot
        self.computed_data["csp"] = cs0 + adot
        self.computed_data["csm"] = cs0 - adot

        # For help with artificial viscosity
        self.computed_data["Q"] = Q

    def cfl_check(self, safety):
        """Check the CFL condition and return the max step size allowed"""
        # Get the propagation speeds
        self.get_info()
        csp = np.abs(self.computed_data["csp"])
        csm = np.abs(self.computed_data["csm"])

        # Find the maximum timesteps for left/right movers
        sm = np.min(self.rdiff / csp)
        sp = np.min(self.rdiff / csm)

        # Compute the maximum timestep
        return min(sm, sp) * safety

    def hasblackhole(self):
        """Returns True if an apparent horizon is detected else False"""
        # Get the horizon condition and u
        self.get_info()
        horizon = self.computed_data["horizon"]
        u = self.computed_data["u"]

        # Go and check everything
        for i, val in enumerate(horizon):
            if val >= 1 and u[i] < 0:
                # Apparent horizon detected
                return True

        # All clear
        return False

    def compute_data(self, xi, um):
        """
        Computes u, m, r, rho, ephi, gamma2, dmdr, dudr, rdot, Q, P and dP
        """
        # Get R, u and m
        r = self.r
        u = um[0:self.gridpoints]
        m = um[self.gridpoints:]

        # Compute derivatives
        dmdr = self.diff.dydx(m)
        dudr = self.diff.dydx(u)

        # Compute various auxiliary variables
        rho = m + r * dmdr / 3
        if np.any(rho < 0):
            raise NegativeDensity()
        gamma2 = exp(xi) + r*r*(u*u-m)

        # Compute drho/dr
        # Note that as this involves a second derivative, it can't be evaluated at
        # the end of the domain, which must be fixed by a boundary condition
        # The last element of drho is zero.
        drho = self.diff.rhoderiv(m)

        # Deal with artificial viscosity: compute Q
        triggered = False
        Q = np.zeros_like(rho)
        if self.viscosity:
            # Construct the triggering condition
            # R' U < - R U'
            # or
            # U + R dU/dR < 0
            dru = u + r * dudr
            test = dru < 0
            if np.any(dru < 0):
                triggered = True
                Q = 1.0 * test  # This converts from true/false to 1/0
                # This is slow, but probably still better than a for loop on true values...
                Q *= self.viscosity * exp(-xi)
                Q *= dru * dru * self.rdiff * self.rdiff

                # We now want to smooth Q a bit, using a haystack filter
                actualQ = Q.copy()
                actualQ *= 10/16
                actualQ[:-1] += Q[1:] * 5/16
                actualQ[1:] += Q[:-1] * 5/16
                actualQ[:-2] += Q[2:]/16
                actualQ[2:] += Q[:-2]/16
                Q = actualQ

                P = rho * (1/3 + Q)
                dQ = self.diff.dydx(Q)
                dP = drho * (1/3 + Q) + rho * dQ
                # Now compute phi
                dphi = - dP / (rho + P)
                phi = np.zeros_like(rho)
                phi[-1] = - 0.25 * log(rho[-1])  # Outer boundary condition
                # And integrate
                # Currently, a stupid trapezoid. Can improve later.
                for i in range(self.gridpoints - 2, -1, -1):
                    phi[i] = phi[i+1] - (dphi[i] + dphi[i + 1]) * self.rdiff[i + 1] / 2
                ephi = np.exp(phi)

        if not triggered:
            # We can use P = w rho
            P = rho / 3
            dP = drho / 3
            ephi = np.power(rho, -1/4)

        # Compute rdot, which is needed for the characteristic speeds
        rdot = r*(u*ephi-1)/2

        # Return the results
        return u, m, r, rho, ephi, gamma2, dmdr, dudr, rdot, Q, P, dP

    def derivs(self, um, xi, params=None):
        """Computes derivatives for evolution"""
        # Compute all the variables about the present state
        u, m, r, rho, ephi, gamma2, dmdr, dudr, rdot, Q, P, dP = self.compute_data(xi, um)

        # These are the equations of motion (time derivatives of m, u)
        mdot = 2*m - 1.5*u*ephi*(P + m)
        udot = u - 0.5*ephi*(0.5*(2*u*u+m+3*P) + gamma2*dP/(rho+P)/r)

        # Convert to Eulerian coordinates
        mdot -= dmdr * rdot
        udot -= dudr * rdot

        # Hack for the origin
        alpha = 0.1
        udot[0] += alpha * ((m[0]-1)/2 - 2 * (u[0] - 1))

        # Boundary condition on U:
        # \dot{U} = - c_s U' + (1/2 - 2 c_s/R) (U - 1)
        # cs = exp(0.5 * xi) / sqrt(12)  # This is the linear speed of sound
        cs = 0.5 / sqrt(3) * ephi[-1] * sqrt(gamma2[-1])  # This is the nonlinear speed of sound
        lastr = r[-1]
        lastr2 = lastr * lastr
        cs2 = cs * cs
        denom = lastr * (2 * cs + lastr)

        alpha = cs
        beta = (12 * cs2 + 6 * cs * lastr + lastr2) / denom / 2
        gamma = -cs * (3 * cs2 + 3 * cs * lastr + lastr2) / lastr / denom
        delta = - cs * (2 * cs2 + 2 * cs * lastr + lastr2) / 2 / denom
        udot[-1] = (- alpha * dudr[-1]
                    - beta * (u[-1] - 1)
                    - gamma * (m[-1] - 1)
                    - delta * dmdr[-1])

        return np.concatenate((udot, mdot))

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
            "xi",       # 14
            "Q",        # 15
        ]
        fulldata = [self.computed_data[name] for name in datanames]
        file.write("# " + "\t".join(map(str, datanames)) + "\n")

        # Go and write the block of data
        for i in range(self.gridpoints):
            dat = [data[i] if isinstance(data, np.ndarray) else data for data in fulldata]
            file.write("\t".join(map(str, dat)) + "\n")
        file.write("\n")
        file.flush()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Misner-Sharp Evolution
"""
from math import exp, pi, sqrt, log
import numpy as np
from dopri5 import DOPRI5, DopriIntegrationError
from newderivs import Derivative
from fancyderivs import Derivative as Deriv4
np.seterr(all='raise', under='ignore')

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

    def __init__(self, r, u, m, xi0=0.0, viscosity=2.0, debug=False,
                 eulerian=True, bhcheck=True, safety=0.75):
        """
        Initialize the data object
        Also sets up the integrator and differentiator
        """
        self.debug = debug
        self.viscosity = viscosity
        self.eulerian = eulerian
        self.bhcheck = bhcheck
        self.safety = safety

        # Store the grid and initial data
        self.r = r
        self.u = u
        self.m = m
        self.gridpoints = len(r)
        self.computed_data = {
            "xi": -1,
            "index": np.array([i for i in range(self.gridpoints)]),
        }
        initialdata = self.construct(r, u, m)

        # Set up the integrator
        self.integrator = DOPRI5(t0=xi0, init_values=initialdata, derivs=self.derivs,
                                 rtol=1e-8, atol=1e-8)

        # Set up the differentiators
        # Note that this is different for Eulerian and Lagrangian evolution
        if eulerian:
            # Set up the differentiator for derivatives with respect to r
            self.diff = Derivative(self.r)
            # Used for artificial viscosity only
            self.diff4 = Deriv4(4)
            self.diff4.set_x(self.r, boundary=-1)
            # Compute the grid spacing (used for CFL checks and artificial viscosity)
            self.rdiff = np.concatenate((np.array([self.r[0] * 2]), np.diff(self.r)))
        else:
            # Set up the differentiator for derivatives with respect to index
            # Note that we make the -1 grid point effectively at -0.5
            self.diff = Derivative(np.array([i + 0.5 for i in range(self.gridpoints)],
                                            dtype=np.dtype('float64')))

        # Turn off Q at the outer boundary
        indices = np.array([i for i in range(self.gridpoints)], dtype=np.dtype('float64'))
        turnover = self.gridpoints - 40
        width = 2
        self.Qenvelope = 1 / (np.exp((indices - turnover) / width) + 1)
        # self.Qenveloped = 1

        # Did the central density ever hit 50?
        self.hit50 = False
        # Has formation stalled? This is set when central density decreases after hitting 50
        self.stalled = False

    def construct(self, r, u, m):
        """
        Construct integrator data based on whether we're doing Lagrangian or
        Eulerian evolution. Can also be used for time derivatives.
        """
        if self.eulerian:
            return np.concatenate((u, m))
        else:
            return np.concatenate((r, u, m))

    def deconstruct(self, data):
        """
        Deconstruct data received from the integrator based on whether we're doing
        Lagrangian or Eulerian evolution. Results are stored in self.u, self.m and self.r
        """
        if self.eulerian:
            self.u = data[0:self.gridpoints]
            self.m = data[self.gridpoints:]
        else:
            self.r = data[0:self.gridpoints]
            self.u = data[self.gridpoints:2*self.gridpoints]
            self.m = data[2*self.gridpoints:]

    def compute_data(self, xi, integratordata=None):
        """
        Computes u, m, r, rho, ephi, gamma2, dmdr, dudr, rdot, Q, P and dP
        """
        # Get R, u and m
        if integratordata is not None:
            # We've been given data from the integrator
            # Need to read it appropriately
            if self.eulerian:
                r = self.r
                u = integratordata[0:self.gridpoints]
                m = integratordata[self.gridpoints:]
            else:
                r = integratordata[0:self.gridpoints]
                u = integratordata[self.gridpoints:2*self.gridpoints]
                m = integratordata[2*self.gridpoints:]
        else:
            # Just use the data stored in the class
            r = self.r
            u = self.u
            m = self.m

        # Compute derivatives
        if self.eulerian:
            dmdr = self.diff.dydx(m, even=True)
            dudr = self.diff.dydx(u, even=False)
        else:
            # Construct these from dm/dA * dA/dr etc
            drda = self.diff.dydx(r, even=False)
            dmdr = self.diff.dydx(m, even=True) / drda
            duda = self.diff.dydx(u, even=False)
            dudr = duda / drda

        # Compute rho
        rho = m + r * dmdr / 3
        if np.any(rho < 0):
            raise NegativeDensity()

        # Construct gamma^2
        gamma2 = exp(xi) + u*u - r*r*m

        # Compute drho/dr
        # Note that as this involves a second derivative, it can't be evaluated at
        # the end of the domain, which must be fixed by a boundary condition
        # The last element of drho is zero.
        if self.eulerian:
            drho = self.diff.rhoderiv(m)
        else:
            drho = self.diff.rhoderiv_lagrange(m, r)

        # Compute e^\phi; we'll modify it later if viscosity is in play
        ephi = np.power(rho, -1/4)

        # Deal with artificial viscosity: compute Q
        triggered = False
        Q = np.zeros_like(rho)
        dQdr = np.zeros_like(rho)
        if self.viscosity:
            if self.eulerian:
                newdudr = self.diff4.dydx(u)
                # Construct the triggering condition
                # U' < 0
                test = newdudr < 0
                if np.any(newdudr < 0):
                    triggered = True
                    Qbool = 1.0 * test  # This converts from true/false to 1/0
                    # This is slow, but probably still better than a for loop on true values...
                    Q = Qbool * self.viscosity * exp(-xi)
                    Q *= newdudr * newdudr * self.rdiff * self.rdiff

                    # We now want to smooth Q a bit, using a haystack filter
                    Qref = Q.copy()
                    # haystack = np.array([10, 5, 1]) / 22
                    haystack = np.array([8, 7, 6, 4, 2, 1]) / 48
                    Q *= haystack[0]
                    for i in range(1, len(haystack)):
                        Q[:-i] += Qref[i:] * haystack[i]
                        Q[i:] += Qref[:-i] * haystack[i]

                    # Turn off Q at the outer boundary
                    Q *= self.Qenvelope

                    dQdr = self.diff.dydx(Q, even=True)
            else:
                # Construct the triggering condition
                # \partial_A U < 0
                test = duda < 0
                if np.any(test):
                    triggered = True
                    Qbool = 1.0 * test  # This converts from true/false to 1/0
                    # This is slow, but probably still better than a for loop on true values...
                    Q = Qbool * self.viscosity * exp(-xi)
                    Q *= duda * duda  # Note that DeltaA = 1

                    # We now want to smooth Q a bit, using a haystack filter
                    Qref = Q.copy()
                    haystack = np.array([10, 5, 1]) / 22
                    haystack = np.array([8, 7, 6, 4, 2, 1]) / 48

                    Q *= haystack[0]
                    for i in range(1, len(haystack)):
                        Q[:-i] += Qref[i:] * haystack[i]
                        Q[i:] += Qref[:-i] * haystack[i]

                    # Turn off Q at the outer boundary
                    Q *= self.Qenvelope

                    dQda = self.diff.dydx(Q, even=True)
                    dQdr = dQda / drda

            if triggered:
                P = rho * (1/3 + Q)
                dP = drho * (1/3 + Q) + rho * dQdr

                # We've already computed rho^(-1/4)
                # Compute the correction factor from Q
                # Note - constructed same way as above, so that
                # when Q and dQdr are zero, X = 0
                X = (drho * 1/3) / (rho/3 + rho) - dP / (P + rho)

                # Integrate X inwards (trapezoid rule)
                Xint = np.zeros_like(X)
                # Construct (positive) interval values for trapezoid rule
                if self.eulerian:
                    deltas = self.rdiff
                else:
                    deltas = np.concatenate((np.array([r[0] * 2]), np.diff(r)))
                # Outer boundary has Xint = 0 by boundary condition
                for i in range(self.gridpoints - 2, -1, -1):
                    Xint[i] = Xint[i+1] - (X[i] + X[i + 1]) * deltas[i + 1] / 2

                # Reconstruct ephi
                ephi *= np.exp(-Xint)

        # Compute pressure and gradients
        P = rho / 3
        dP = drho / 3

        # Compute rdot
        # Needed for evolution (Lagrangian) and characteristic speeds (Eulerian)
        rdot = (u*ephi-r)/2

        # Return the results
        return u, m, r, rho, ephi, gamma2, dmdr, dudr, rdot, Q, P, dP, dQdr

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
        u, m, r, rho, ephi, gamma2, _, _, cs0, Q, _, _, _ = self.compute_data(xi)
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
        self.computed_data["ufull"] = a * H * u
        self.computed_data["mfull"] = 4 * pi / 3 * rhob * np.power(rfull, 3) * m

        # Horizon condition
        self.computed_data["horizon"] = r * r * m * H

        # Compute characteristic speeds
        adot = 0.5 / sqrt(3) * ephi * gamma
        if self.eulerian:
            # Three speeds of characteristics: dR/dxi
            self.computed_data["cs0"] = cs0  # rdot
            self.computed_data["csp"] = cs0 + adot
            self.computed_data["csm"] = cs0 - adot
        else:
            # Three speeds of characteristics: dR/dxi
            self.computed_data["cs0"] = np.zeros_like(adot)
            self.computed_data["csp"] = adot
            self.computed_data["csm"] = - adot

        # For help with artificial viscosity
        self.computed_data["Q"] = Q

    def derivs(self, um, xi, params=None):
        """Computes derivatives for evolution"""
        # Compute all the variables about the present state
        u, m, r, rho, ephi, gamma2, dmdr, dudr, rdot, Q, P, dP, dQdr = self.compute_data(xi, um)

        # These are the equations of motion (time derivatives of m, u)
        mdot = 2*m - 1.5*u*ephi*(P + m)/r
        udot = 0.5*u - 0.5*ephi*(gamma2*dP/(rho+P) + 0.5*(m+3*P)*r)

        # Boundary condition on U:
        # cs = 1 / sqrt(12) * exp(xi/2)  # This is the linear speed of sound
        cs = 1 / sqrt(12) * ephi[-1] * sqrt(gamma2[-1])  # This is the nonlinear speed of sound
        lastr = r[-1]
        lastr2 = lastr * lastr
        cs2 = cs * cs
        denom = lastr * (2 * cs + lastr)

        lastdmdr = dmdr[-1]
        lastdudr = dudr[-1]

        W = - cs
        X = - (12 * cs2 + 6 * cs * lastr + lastr2) / denom / 2
        Z = cs * (3 * cs2 + 3 * cs * lastr + lastr2) / lastr / denom
        Y = cs * (2 * cs2 + 2 * cs * lastr + lastr2) / 2 / denom

        udot[-1] = (
                    + W * (lastdudr - u[-1] / lastr)
                    + X * (u[-1] - lastr)
                    + Y * lastr * lastdmdr
                    + Z * lastr * (m[-1] - 1)
                    + u[-1] * rdot[-1] / r[-1]
                    )

        # # Check if we need a viscosity term
        # if self.viscosity:
        #     # TODO: Check this in Eulerian case
        #     # Best vehicle: Amax = 15, amp = 1.50
        #     if Q[-1]:
        #         # Add Q into the 3 \tilde{P} term (see Eq. 52c)
        #         udot[-1] += - 1/2 * ephi[-1] * lastr/2 * 3 * rho[-1] / 3 * Q[-1]
        #     if dQdr[-1]:
        #         # Add Q' and Q into the \tilde{P}' term (also Eq. 52c)
        #         udot[-1] += - 1/2 * ephi[-1] * gamma2[-1] / rho[-1] / (1 + 1/3 + Q[-1]) * dQdr[-1]

        # Return the appropriate results
        if self.eulerian:
            # Convert to Eulerian coordinates
            # Make sure to do this *after* computing the boundary condition!
            mdot -= dmdr * rdot
            udot -= dudr * rdot
            return self.construct(None, udot, mdot)
        else:
            return self.construct(rdot, udot, mdot)

    def step(self, newtime):
        """Takes a step forwards in time"""
        stepcount = 0
        lastrho = 0
        while self.integrator.t < newtime:
            stepcount += 1
            try:
                self.integrator.step(newtime)
            except DopriIntegrationError as err:
                raise IntegrationError(err.args[0])

            # Store result
            self.deconstruct(self.integrator.values)

            # Check if a black hole is present
            if self.bhcheck and self.hasblackhole():
                raise BlackHoleFormed()

            # Change CFL condition
            self.integrator.update_max_h(self.cfl_check())

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

    def cfl_check(self):
        """Check the CFL condition and return the max step size allowed"""
        # Get the propagation speeds
        self.get_info()
        if self.eulerian:
            csp = np.abs(self.computed_data["csp"])
            csm = np.abs(self.computed_data["csm"])

            # Find the maximum timesteps for left/right movers
            sm = np.min(self.rdiff / csp)  # TODO: Can this rdiff be precomputed and stored?
            sp = np.min(self.rdiff / csm)

            # Compute the maximum timestep
            return min(sm, sp) * self.safety
        else:
            csp = self.computed_data["csp"]
            rdiff = np.concatenate((np.array([self.r[0] * 2]), np.diff(self.r)))
            return np.min(rdiff / csp) * self.safety

    def hasblackhole(self):
        """Returns True if an apparent horizon is detected else False"""
        # Get the horizon condition and u
        self.get_info()
        horizon = self.computed_data["horizon"]
        u = self.computed_data["u"]

        # Go and check everything
        # FIXME This can probably be done more efficiently using numpy...
        for i, val in enumerate(horizon):
            if val >= 1 and u[i] < 0:
                # Apparent horizon detected
                return True

        # All clear
        return False

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
            "ephi",     # 16
        ]
        fulldata = [self.computed_data[name] for name in datanames]
        file.write("# " + "\t".join(map(str, datanames)) + "\n")

        # Go and write the block of data
        for i in range(self.gridpoints):
            dat = [data[i] if isinstance(data, np.ndarray) else data for data in fulldata]
            file.write("\t".join(map(str, dat)) + "\n")
        file.write("\n")
        file.flush()

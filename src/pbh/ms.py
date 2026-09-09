"""Misner-Sharp black hole evolution.

Provides the :class:`MS` evolver together with the Lagrangian (:class:`MSLagrangian`) and Eulerian
(:class:`MSEulerian`) implementations of the equations of motion. Equation numbers refer to the paper in
``paper.pdf``.
"""

import math
from abc import ABC, abstractmethod

import numpy as np

from pbh.base import BlackHoleEvolver, EOMHandler, EvolverError, FloatArray, Status, cached_property
from pbh.derivs import Derivative


class MS(BlackHoleEvolver["MSCommon"]):
    """Evolver class for Misner-Sharp evolution.

    Adds the following features:

    * Black hole detection, with option to abort at detection if ``black_hole_check`` is True.
    * Timeout detection, which aims to detect if a black hole is unlikely to form and abort evolution.
      Starts checking at the point when the longest wavelength mode first peaks. If ``enforce_timeout`` is True,
      automatically stops at this stage (useful for linear evolutions).

    Both features rely only on local quantities, and so are independent of EOM implementation details.
    """

    def __init__(
        self,
        eomhandler: type[MSCommon],
        black_hole_check: bool = True,
        enforce_timeout: bool = False,
        rtol: float = 1e-8,
        atol: float = 1e-8,
        cfl_safety: float = 0.75,
        viscosity: float | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize the evolver.

        Args:
            eomhandler: The equation of motion handler class (:class:`MSEulerian` or :class:`MSLagrangian`).
            black_hole_check: Whether to stop evolution once an apparent horizon forms.
            enforce_timeout: Whether to stop unconditionally once the timeout time is reached.
            rtol: Relative tolerance for the integrator.
            atol: Absolute tolerance for the integrator.
            cfl_safety: Safety factor applied to the CFL step size limit.
            viscosity: Artificial viscosity coefficient (None or 0 to disable).
            debug: Whether to print debugging information.
        """
        super().__init__(
            eomhandler=eomhandler, rtol=rtol, atol=atol, cfl_safety=cfl_safety, viscosity=viscosity, debug=debug
        )
        self.black_hole_check = black_hole_check
        self.enforce_timeout = enforce_timeout
        self.timeouttime = math.inf
        if self.debug:
            print("Initializing MS evolver")

    def set_initial_conditions(self, start_xi: float, *start_fields: FloatArray) -> None:
        """Set initial conditions, and compute the timeout time from the grid extent."""
        super().set_initial_conditions(start_xi, *start_fields)

        # Set timeout to be the time at which the longest wavelength mode peaks for the first time.
        self.timeouttime = 0.8278 + 2 * math.log(self.eomhandler.r[-1])  # Eq. (90)
        # Note that this works for both Eulerian and Lagrangian codes, as A and R are interchangeable at linear order.

    def post_step_processing(self) -> bool:
        """Check after every step whether a black hole has formed."""
        return self.black_hole_check and self.black_hole_formed()

    def black_hole_formed(self) -> bool:
        """Return True if an apparent horizon has formed, else False."""
        # Set EOM handler to use the appropriate field values
        self.eomhandler.set_fields(self.integrator.t, self.integrator.values)

        # Extract relevant values
        horizon = self.eomhandler.horizon
        u = self.eomhandler.u

        if np.any(np.logical_and(horizon >= 1, u < 0)):  # Eq. (52)
            # Apparent horizon detected
            self.status = Status.BLACKHOLE_FORMED
            return True

        # No apparent horizon
        return False

    def post_output_processing(self) -> bool:
        """Check after every output whether a black hole is unlikely to form at this stage."""
        # Set EOM handler to use the appropriate field values
        self.eomhandler.set_fields(self.integrator.t, self.integrator.values)

        # Check if a black hole is unlikely to form
        if self.xi > self.timeouttime:
            # Check to see if we're not close to black hole formation
            m = self.eomhandler.m
            if self.enforce_timeout or (np.all(m > 0.5) and np.all(m < 1.5)):
                self.status = Status.TIMEOUT
                return True

        return False

    def debug_evolve_complete(self, stepcount: int) -> None:
        """Print step statistics at the completion of the evolve method."""
        print(f"MS: Stepped to xi = {self.xi:.5f} in {stepcount} steps, last stepsize was {self.integrator.hdid:.5f}")


class MSCommon(EOMHandler, ABC):
    """Common parts of the Lagrangian/Eulerian equations of motion for Misner-Sharp evolution.

    Parts that must be specifically implemented by subclasses are introduced as abstract methods.
    """

    def __init__(self, parent: BlackHoleEvolver[MSCommon]) -> None:
        super().__init__(parent)
        # Storage for differentiation object (set by initialize_derivatives)
        self._diff: Derivative | None = None

    @property
    def diff(self) -> Derivative:
        """The finite difference operator. Only available once initialize_derivatives has been called."""
        if self._diff is None:
            raise ValueError("Must call initialize_derivatives before differentiating")
        return self._diff

    @property
    @abstractmethod
    def dmdr(self) -> FloatArray:
        r"""d\bar{m}/d\bar{R}."""

    @property
    @abstractmethod
    def dudr(self) -> FloatArray:
        r"""d\bar{U}/d\bar{R}."""

    @cached_property
    def rho(self) -> FloatArray:
        r"""\bar{\rho}. Raises EvolverError (setting status NEGATIVE_ENERGY_DENSITY) if negative anywhere."""
        rho = self.m + self.r * self.dmdr / 3  # Eq. (44d)
        if np.any(rho < 0):
            self._parent.status = Status.NEGATIVE_ENERGY_DENSITY
            raise EvolverError()
        return rho

    @property
    @abstractmethod
    def drhodr(self) -> FloatArray:
        r"""d\bar{\rho}/d\bar{R}."""

    @property
    @abstractmethod
    def rdiff(self) -> FloatArray:
        """The distance between successive gridpoints, including r[-1] = -r[0]."""

    @cached_property
    def viscosity_present(self) -> bool:
        """Whether or not artificial viscosity is currently triggered."""
        self._computeQ()
        return self.viscosity_present

    @cached_property
    def Q(self) -> FloatArray:
        r"""\bar{Q}, the artificial viscosity."""
        self._computeQ()
        return self.Q

    @cached_property
    def dQdr(self) -> FloatArray:
        r"""d\bar{Q}/d\bar{R}."""
        self._computeQ()
        return self.dQdr

    @abstractmethod
    def _computeQ(self) -> None:
        """Compute the artificial viscosity Q and its derivative, storing them with :meth:`_cacheQ`."""

    def _cacheQ(self, viscosity_present: bool, Q: FloatArray, dQdr: FloatArray) -> None:
        """Store the artificial viscosity results in the cache."""
        if self._cache is None:
            raise ValueError("Must initialize EOMHandler with set_fields before requesting quantities")
        self._cache["viscosity_present"] = viscosity_present
        self._cache["Q"] = Q
        self._cache["dQdr"] = dQdr

    @staticmethod
    def haystack(input_array: FloatArray) -> FloatArray:
        """Apply a haystack filter to smooth an array."""
        result = input_array.copy()
        haystack = np.array([8, 7, 6, 4, 2, 1]) / 48
        result *= haystack[0]
        for i in range(1, len(haystack)):
            result[:-i] += input_array[i:] * haystack[i]
            result[i:] += input_array[:-i] * haystack[i]
        return result

    @cached_property
    def ephi(self) -> FloatArray:
        r"""e^\phi."""
        ephi_analytic = np.power(self.rho, -1 / 4)  # Eq. (45)

        if not self.viscosity_present:
            return ephi_analytic

        # Compute the correction due to artificial viscosity.
        # This following is constructed so that when Q and dQdr are vanishing, we get exact cancellations at machine
        # precision (the first term is constructed exactly the same way as the last term).
        x = (self.drhodr / 3) / (self.rho / 3 + self.rho) - self.dPdr / (self.P + self.rho)  # Eq. (210-211)

        # Integrate X inwards using the trapezoid rule.
        # Construct (positive) interval values for trapezoid rule, using xint=0 (boundary condition) at outer boundary.
        trap_integrand = (x[:-1] + x[1:]) * self.rdiff[1:] / 2
        xint = np.cumsum(trap_integrand)  # This performs the integation
        xint = np.insert(xint, 0, 0)  # Add a 0 to the start of the cumulative sum
        xint -= xint[-1]  # Add a constant to apply the boundary condition

        # Reconstruct e^phi
        return ephi_analytic * np.exp(-xint)  # Eq. (211)

    @cached_property
    def P(self) -> FloatArray:
        r"""\bar{P}."""
        if self.viscosity_present:
            # This is written in such a way to help with machine precision cancellations when self.Q is 0
            return self.rho / 3 + self.rho * self.Q  # Eq. (41b)
        return self.rho / 3  # Eq. (41b)

    @cached_property
    def dPdr(self) -> FloatArray:
        r"""d\bar{P}/d\bar{R}."""
        if self.viscosity_present:
            # This is written in such a way to help with machine precision cancellations when self.Q is 0
            return self.drhodr / 3 + self.drhodr * self.Q + self.rho * self.dQdr  # Eq. (41b)
        return self.drhodr / 3  # Eq. (41b)

    @cached_property
    def c_characteristic(self) -> FloatArray:
        r"""The characteristic speed d\bar{R}/d\xi (where \bar{R} is a characteristic position, not a field)."""
        return self.ephi * self.gamma / np.sqrt(12)  # Eq. (200)

    @cached_property
    def c_fluid(self) -> FloatArray:
        r"""The fluid velocity d\bar{R}/d\xi (where \bar{R} is a field)."""
        return (self.u * self.ephi - self.r) / 2  # Eq. (44a)

    def lagrangian_derivatives(self, params: object) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Evaluate the Lagrangian equations of motion, returning a tuple (rdot, udot, mdot)."""
        del params  # Unused
        # Extract the required quantities
        u = self.u
        m = self.m
        r = self.r
        rho = self.rho
        ephi = self.ephi
        P = self.P
        gamma2 = self.gamma2
        dPdr = self.dPdr

        # Here are the equations of motion
        rdot = self.c_fluid
        mdot = 2 * m - 1.5 * u * ephi * (P + m) / r  # Eq. (44b)
        udot = (u - ephi * (gamma2 * dPdr / (rho + P) + 0.5 * (m + 3 * P) * r)) / 2  # Eq. (44c)

        # We now impose the outer boundary condition on U
        # cs = exp(xi/2) / sqrt(12)  # This is the linear speed of sound, which can be read from Eq. (59)
        cs = self.c_characteristic[-1]  # This is the **nonlinear** speed of sound
        r0 = r[-1]
        r02 = r0 * r0
        cs2 = cs * cs
        denom = r0 * (2 * cs + r0)
        # The following implement Eqs. (101a-d)
        W = -cs
        X = -(12 * cs2 + 6 * cs * r0 + r02) / 2 / denom
        Y = cs * (2 * cs2 + 2 * cs * r0 + r02) / 2 / denom
        Z = cs * (3 * cs2 + 3 * cs * r0 + r02) / r0 / denom
        udot[-1] = (
            W * (self.dudr[-1] - u[-1] / r0)  # Eq. (102)
            + X * (u[-1] - r0)
            + Y * r0 * self.dmdr[-1]
            + Z * r0 * (m[-1] - 1)
            + u[-1] * rdot[-1] / r[-1]
        )

        return rdot, udot, mdot


class MSLagrangian(MSCommon):
    r"""Lagrangian implementation of the equations of motion for the Misner-Sharp evolution.

    We evolve \bar{m}(A), \bar{U}(A) and \bar{R}(A), where A is the radial coordinate that moves with the fluid.
    """

    def initialize_derivatives(self) -> None:
        """Set up the differentiator for derivatives with respect to index.

        The first gridpoint is at 0.5, then 1.5, etc. This differs from the index values by 0.5.
        This needs to be done so that the differentiator places the symmetry gridpoint at -0.5.
        """
        self._diff = Derivative((self._parent.index + 0.5).astype(np.float64))

    @cached_property
    def dmda(self) -> FloatArray:
        r"""d\bar{m}/dA."""
        return self.diff.dydx(self.m, even=True)  # By definition - derivative w.r.t A

    @cached_property
    def drda(self) -> FloatArray:
        r"""d\bar{R}/dA."""
        return self.diff.dydx(self.r, even=False)  # By definition - derivative w.r.t A

    @cached_property
    def duda(self) -> FloatArray:
        r"""d\bar{U}/dA."""
        return self.diff.dydx(self.u, even=False)  # By definition - derivative w.r.t A

    @cached_property
    def dmdr(self) -> FloatArray:
        r"""d\bar{m}/d\bar{R}."""
        return self.dmda / self.drda  # By definition - derivative w.r.t \bar{R}

    @cached_property
    def dudr(self) -> FloatArray:
        r"""d\bar{U}/d\bar{R}."""
        return self.duda / self.drda  # By definition - derivative w.r.t \bar{R}

    @cached_property
    def drhodr(self) -> FloatArray:
        r"""d\bar{\rho}/d\bar{R}.

        Note that this is computed from \bar{m} and \bar{R} directly, rather than taking another difference operation.
        As a second derivative, it can't be evaluated at the end of the domain, which must be fixed by a boundary
        condition instead. The last element is returned as zero.
        """
        return self.diff.rhoderiv_lagrange(self.m, self.r)  # Eq. (199)

    def _computeQ(self) -> None:
        """Compute the artificial viscosity Q and its derivative."""
        # Trigger if viscosity is nonzero and \partial_A U < 0
        test = self.duda < 0  # Eq. (208b), note an old definition for \bar{U}
        viscosity = self._parent.viscosity
        if not viscosity or not np.any(test):
            self._cacheQ(False, np.zeros_like(self.r), np.zeros_like(self.r))
            return

        # Construct Q (note that DeltaA = 1)
        Q = (
            test * viscosity * self.H * self.duda * self.duda
        )  # Eqs. (207) and (43c), note an old definition for \bar{U}

        # Smooth Q a bit (which has a binary on/off switch from test that is discontinuous)
        Q = self.haystack(Q)

        # Apply the Q envelope to turn off Q at the outer boundary
        Q *= self._parent.Qenvelope

        # Cache Q and dQdr
        dQda = self.diff.dydx(Q, even=True)
        self._cacheQ(True, Q, dQda / self.drda)

    @cached_property
    def cs0(self) -> FloatArray:
        """The fluid speed c_0."""
        # This is just zero in Lagrangian coordinates, where our coordinates move with the fluid elements
        return np.zeros_like(self.r)

    @cached_property
    def csp(self) -> FloatArray:
        r"""The speed of sound c_+ = d\bar{R}/d\xi."""
        return self.c_characteristic

    @cached_property
    def csm(self) -> FloatArray:
        r"""The speed of sound c_- = -d\bar{R}/d\xi."""
        return -self.c_characteristic

    def derivatives(self, params: object) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return a tuple of time derivatives for evolution (rdot, udot, mdot)."""
        return self.lagrangian_derivatives(params)

    def cfl_step(self) -> float:
        """Return the maximum step size allowed by the CFL condition."""
        # Compute the travel time between gridpoints
        traveltime = self.rdiff / self.csp
        # Find the minimum travel time
        return float(np.min(traveltime))

    @cached_property
    def rdiff(self) -> FloatArray:
        """The distance between gridpoints, using 2*r[0] to represent the distance across the origin."""
        return np.concatenate((np.array([self.r[0] * 2]), np.diff(self.r)))


class MSEulerian(MSCommon):
    r"""Eulerian implementation of the equations of motion for the Misner-Sharp evolution.

    We evolve \bar{m}(R) and \bar{U}(R), using a fixed grid in R (circumferential radius).
    """

    def __init__(self, parent: BlackHoleEvolver[MSCommon]) -> None:
        super().__init__(parent)
        self._rdiff: FloatArray | None = None

    def initialize_derivatives(self) -> None:
        """Initialize derivative operator, as well as the rdiff value."""
        self._diff = Derivative(self.r)
        self._rdiff = np.concatenate((np.array([self.r[0] * 2]), np.diff(self.r)))

    @property
    def rdiff(self) -> FloatArray:
        """The distance between gridpoints, using 2*r[0] to represent the distance across the origin."""
        if self._rdiff is None:
            raise ValueError("Must call initialize_derivatives before requesting rdiff")
        return self._rdiff

    @cached_property
    def dmdr(self) -> FloatArray:
        r"""d\bar{m}/d\bar{R}."""
        return self.diff.dydx(self.m, even=True)  # By definition - derivative w.r.t \bar{R}

    @cached_property
    def dudr(self) -> FloatArray:
        r"""d\bar{U}/d\bar{R}."""
        return self.diff.dydx(self.u, even=False)  # By definition - derivative w.r.t \bar{R}

    @cached_property
    def drhodr(self) -> FloatArray:
        r"""d\bar{\rho}/d\bar{R}.

        This is computed directly as a second derivative. As such, it can't be evaluated at the outer boundary,
        which must be fixed by a boundary condition. The last element is thus returned as 0.
        """
        return self.diff.rhoderiv(self.m)  # Eq. (199)

    def _computeQ(self) -> None:
        """Compute the artificial viscosity Q and its derivative."""
        # Trigger if viscosity is nonzero and dudr < 0
        test = self.dudr < 0  # Eq. (208b), note an old definition for \bar{U}
        viscosity = self._parent.viscosity
        if not viscosity or not np.any(test):
            self._cacheQ(False, np.zeros_like(self.r), np.zeros_like(self.r))
            return

        # Construct Q
        deltau = self.dudr * self.rdiff
        # Eqs. (207) and (43c), note an old definition for \bar{U}
        Q = test * viscosity * np.exp(-self.xi) * deltau * deltau

        # Smooth Q a bit (which has a binary on/off switch from test that is discontinuous)
        Q = self.haystack(Q)

        # Apply the Q envelope to turn off Q at the outer boundary
        Q *= self._parent.Qenvelope

        # Cache Q and dQdr
        self._cacheQ(True, Q, self.diff.dydx(Q, even=True))

    @cached_property
    def cs0(self) -> FloatArray:
        r"""The fluid speed d\bar{R}/d\xi."""
        return self.c_fluid

    @cached_property
    def csp(self) -> FloatArray:
        r"""The characteristic speed d\bar{R}/d\xi."""
        return self.c_fluid + self.c_characteristic

    @cached_property
    def csm(self) -> FloatArray:
        r"""The characteristic speed -d\bar{R}/d\xi."""
        return self.c_fluid - self.c_characteristic

    def derivatives(self, params: object) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return a tuple of time derivatives for evolution (rdot, udot, mdot)."""
        # Construct the Lagrangian EOMs
        rdot, udot, mdot = self.lagrangian_derivatives(params)
        # Convert to Eulerian EOMs. From the chain rule:
        # d/d\xi|A = d/d\xi|R + dR/d\xi|A d/dR|\xi
        # So, d/d\xi|R = d/d\xi|A - rdot d/dR|\xi
        mdot -= self.dmdr * rdot
        udot -= self.dudr * rdot
        rdot = np.zeros_like(rdot)
        return rdot, udot, mdot

    def cfl_step(self) -> float:
        """Return the maximum step size allowed by the CFL condition."""
        # Because our speeds of sound are not symmetric (not in the fluid rest frame), we compute a maximum time step
        # in both directions.
        # Compute the travel time between gridpoints
        traveltime_p = self.rdiff / np.abs(self.csp)
        traveltime_m = self.rdiff / np.abs(self.csm)
        # Find the minimum travel time
        return float(min(np.min(traveltime_p), np.min(traveltime_m)))

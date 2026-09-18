"""DOPRI5 integration algorithm.

Dormand-Prince 5(4) embedded Runge-Kutta integrator with adaptive step size control and first-same-as-last (FSAL)
reuse of the final slope evaluation.

Jolyon Bloomfield, October 2017.
"""

import math
from collections.abc import Callable
from typing import Final

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type DerivsFunc = Callable[[float, FloatArray, object], FloatArray]
"""Signature of a derivative function: ``derivs(t, values, params) -> dvalues/dt``."""


class DopriIntegrationError(Exception):
    """Error raised when integration cannot proceed."""


# Butcher tableau for DOPRI5
_TIMES: Final[FloatArray] = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
_COEFFS: Final[tuple[FloatArray, ...]] = (
    np.array([]),
    np.array([1 / 5]),
    np.array([3 / 40, 9 / 40]),
    np.array([44 / 45, -56 / 15, 32 / 9]),
    np.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
    np.array([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]),
    np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]),
)
_ERRORS: Final[FloatArray] = np.array([71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40])


def _linear_combination(coeffs: FloatArray, arrays: list[FloatArray]) -> FloatArray:
    """Return ``sum(c * a for c, a in zip(coeffs, arrays))`` accumulated left to right."""
    result = coeffs[0] * arrays[0]
    for c, a in zip(coeffs[1:], arrays[1:], strict=True):
        result = result + c * a
    return result


class DOPRI5:
    """Dormand-Prince 5th order integrator."""

    def __init__(
        self,
        t0: float,
        init_values: FloatArray,
        derivs: DerivsFunc,
        init_h: float = 0.01,
        min_h: float = 5e-8,
        max_h: float = 1.0,
        rtol: float = 1e-7,
        atol: float = 1e-7,
        params: object = None,
    ) -> None:
        """Initialize the integrator.

        Args:
            t0: Starting time.
            init_values: Starting values.
            derivs: Derivative function, called as ``derivs(t, values, params)``.
            init_h: Initial step size.
            min_h: Minimum step size.
            max_h: Maximum step size.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            params: Parameters passed through to the derivative function.
        """
        self.derivs = derivs
        self.values = init_values
        self.t = t0
        self.hnext = init_h  # Step we're about to take
        self.max_h = max_h
        self.min_h = min_h
        self.rtol = rtol
        self.atol = atol
        self.params = params

        # Internal variables
        self.hdid = 0.0  # Previous step we just took
        self.dxdt: FloatArray | None = None  # Used for FSAL
        self._newvalues: FloatArray = init_values
        self._newdxdt: FloatArray | None = None
        self._errors: FloatArray = np.zeros_like(init_values)

    def set_init_values(self, t0: float, init_values: FloatArray) -> None:
        """Reset the time and values, e.g. to restart an integration."""
        self.values = init_values
        self.t = t0
        self.clear_fsal()

    def update_max_h(self, new_max_h: float) -> None:
        """Update the max step size."""
        if new_max_h < self.min_h:
            raise DopriIntegrationError("Requested max step size less than min step size")
        self.max_h = new_max_h
        self.hnext = min(self.hnext, self.max_h)

    def clear_fsal(self) -> None:
        """Clear FSAL information, forcing it to be recalculated."""
        self.dxdt = None

    def step(self, newtime: float) -> None:
        """Take a single accepted step, never going past ``newtime``."""
        rejected = False

        while True:
            # Comment these two lines out if you want to allow it to go past newtime
            if self.t + self.hnext > newtime:
                self.hnext = newtime - self.t
            self._take_step(self.hnext)
            if self._good_step(rejected):
                break
            rejected = True

        # Update our data
        self.t += self.hdid
        self.values = self._newvalues
        self.dxdt = self._newdxdt

    def _good_step(self, rejected: bool, minscale: float = 0.2, maxscale: float = 5, safety: float = 0.8) -> bool:
        """Check whether the previous step was good, and update the step size.

        Args:
            rejected: Whether we rejected a previous attempt at this step.
            minscale: The minimum factor by which we will scale down the stepsize.
            maxscale: The maximum factor by which we will scale up the stepsize.
            safety: The safety factor in the stepsize estimation.

        Returns:
            True if the step is accepted, else False.
        """
        # Compute the scaled error of the past step
        err = self._error()
        if err <= 1.0:
            # Step was good
            # Figure out how to scale our next step
            if err == 0.0:
                scale = maxscale
            else:
                scale = safety * math.pow(err, -0.2)
                scale = max(scale, minscale)
                scale = min(scale, maxscale)
            # Make sure we're not increasing the step if we just rejected something
            if rejected:
                scale = min(scale, 1.0)
            # Update the step sizes
            self.hdid = self.hnext
            self.hnext = self.hdid * scale
            self.hnext = min(self.hnext, self.max_h)
            self.hnext = max(self.hnext, self.min_h)
            return True

        # Error was too big
        if self.hnext == self.min_h:
            raise DopriIntegrationError("Step size decreased below minimum threshold")
        # Try again!
        scale = max(safety * math.pow(err, -0.2), minscale)
        self.hnext *= scale
        self.hnext = max(self.hnext, self.min_h)
        return False

    def _error(self) -> float:
        """Compute the normalized error in the step just taken."""
        maxed = np.maximum(np.abs(self.values), np.abs(self._newvalues))
        delta = self.atol + self.rtol * maxed
        temp = self._errors / delta
        return math.sqrt(float(np.dot(temp, temp)) / len(temp))

    def _take_step(self, h: float) -> None:
        """Take an individual (trial) step with size h."""
        # Check that we're initialized
        if self.dxdt is None:
            self.dxdt = self.derivs(self.t, self.values, self.params)

        # Compute the slopes and updated positions
        slopes: list[FloatArray] = [self.dxdt]  # stored from previous step
        # The first stage is a single term; (h * c) * slope is used here for bit-for-bit consistency with earlier
        # versions of this code.
        newvals = self.values + h * _COEFFS[1][0] * slopes[0]
        slopes.append(self.derivs(self.t + h * _TIMES[1], newvals, self.params))
        for i in range(2, 7):
            newvals = self.values + h * _linear_combination(_COEFFS[i], slopes)
            slopes.append(self.derivs(self.t + h * _TIMES[i], newvals, self.params))

        # Save the results
        self._newvalues = newvals
        self._newdxdt = slopes[6]
        # Compute the errors
        self._errors = h * _linear_combination(_ERRORS, slopes)

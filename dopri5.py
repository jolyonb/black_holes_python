"""
DOPRI5 Integration Algorithm

Jolyon Bloomfield
October 2017
"""
import numpy as np
import math

class DopriIntegrationError(Exception):
    """Error class for integration"""
    pass

class DOPRI5(object):
    """Dormand-Prince 5th order integrator"""

    def __init__(self,
                 t0,                # Starting time
                 init_values,       # Starting values
                 derivs,            # Derivative function
                 init_h=0.01,       # Initial step size
                 min_h=5e-8,        # Minimum step size
                 max_h=1.0,         # Maximum step size
                 rtol=1e-7,         # Relative tolerance
                 atol=1e-7,         # Absolute tolerance
                 params=None):      # Parameters to pass to the derivatives function
        """
        Initialize the integrator
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
        self.hdid = 0        # Previous step we just took
        self.dxdt = None    # Used for FSAL
        
    def set_init_values(self, t0, init_values):
        self.values = init_values
        self.t = t0

    def update_max_h(self, new_max_h):
        """Updates the max step size"""
        if new_max_h < self.min_h:
            raise DopriIntegrationError("Requested max step size less than min step size")
        self.max_h = new_max_h
        self.hnext = min(self.hnext, self.max_h)

    def clear_fsal(self):
        """Clears FSAL information, forcing it to be recalculated"""
        self.dxdt = None

    def step(self, newtime):
        """Take a step"""
        rejected = False

        while True:
            # Comment these two lines out if you want to allow it to go past newtime
            if self.t + self.hnext > newtime:
                self.hnext = newtime - self.t
            self._take_step(self.hnext)
            if self._good_step(rejected):
                break
            else:
                rejected = True

        # Update our data
        self.t += self.hdid
        self.values = self.newvalues
        self.dxdt = self.newdxdt

    def _good_step(self, rejected, minscale=0.2, maxscale=5, safety=0.8):
        """
        Checks if the previous step was good, and updates the step size
        rejected stores whether or not we rejected a previous attempt at this step
        minscale is the minimum we will scale down the stepsize
        maxscale is the maximum we will scale up the stepsize
        safety is the safety factor in the stepsize estimation
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
        else:
            # Error was too big
            if self.hnext == self.min_h:
                raise DopriIntegrationError("Step size decreased below minimum threshold")
            # Try again!
            scale = max(safety * math.pow(err, -0.2), minscale)
            self.hnext *= scale
            self.hnext = max(self.hnext, self.min_h)
            return False

    def _error(self):
        """Computes the normalized error in the step just taken"""
        maxed = np.column_stack((np.abs(self.values), np.abs(self.newvalues)))
        maxed = np.max(maxed, axis=1)
        delta = self.atol + self.rtol * maxed
        temp = self.errors / delta
        return math.sqrt(np.dot(temp, temp) / len(temp))

    # Coefficients for DOPRI5
    _dopri5times = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    _dopri5coeffs = [0,
                     np.array([1/5]),
                     np.array([3/40, 9/40]),
                     np.array([44/45, -56/15, 32/9]),
                     np.array([19372/6561, -25360/2187, 64448/6561, -212/729]),
                     np.array([9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]),
                     np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
                     ]
    _dopri5errors = np.array([71/57600, 0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40])

    def _take_step(self, h):
        """Take an individual step with size h"""
        # Check that we're initialized
        if self.dxdt is None:
            self.dxdt = self.derivs(self.t, self.values, self.params)

        # Compute the slopes and updated positions
        slopes = [None] * 7
        slopes[0] = self.dxdt   # stored from previous step
        newvals = self.values + h*self._dopri5coeffs[1][0]*slopes[0]
        slopes[1] = self.derivs(self.t + h*self._dopri5times[1], newvals, self.params)
        for i in range(2, 7):
            newvals = self.values + h*sum(self._dopri5coeffs[i][j]*slopes[j] for j in range(i))
            slopes[i] = self.derivs(self.t + h*self._dopri5times[i], newvals, self.params)

        # Save the results
        self.newvalues = newvals
        self.newdxdt = slopes[6]
        # Compute the errors
        self.errors = h * sum(self._dopri5errors[i]*slopes[i] for i in range(7))

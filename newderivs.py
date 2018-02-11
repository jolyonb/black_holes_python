# -*- coding: utf-8 -*-
"""
Computes second order finite difference derivatives for MS evolution
"""
import numpy as np

class DerivativeError(Exception):
    pass

class Derivative(object):
    """
    Computes a finite difference derivatives for even functions on a grid without a
    gridpoint at the origin. In particular, the following derivatives are computed:

    * dy/dx
    * rhoderiv = (x d^2y/dx^2 + 4 dy/dx) / 3 = 4/3 * d/d(x^4) (x^4 dy/dx)
    """

    def __init__(self, xvals):
        """
        Initializes all of the derivative coefficients

        xvals is an array of x values
        """
        # Initialize the stencil storage
        self.length = length = len(xvals)
        if length < 2:
            raise DerivativeError("Grid too short to compute derivatives")

        # Construct the xvals differences
        # diffs[i] is xvals[i] - xvals[i-1], which goes backwards so we can get
        # the boundary spacing correct
        diffs = np.insert(np.diff(xvals), 0, xvals[0]*2)
        invdiffs = 1/diffs
        # doublediffs[i] = x[i+1] - x[i-1]
        doublediffs = diffs[1:] + diffs[:-1]
        invdoublediffs = 1/doublediffs

        # Make the slices that will extract the correct components for dot products
        self.seqs = [slice(i-1, i+2) for i in range(length)]
        self.seqs[0] = slice(0, 3)
        self.seqs[-1] = slice(length-3, length)

        # Construct the simple linear derivative
        # This is a centered nonuniform first order derivative using three gridpoints
        # to be accurate to O(h^2)
        # df/dx = (f[i]-f[i-1])/(x[i]-x[i-1]) + (f[i+1]-f[i])/(x[i+1]-x[i])
        #           + (f[i+1]-f[i-1])/(x[i+1]-x[i-1])
        self._dydxstencil = np.zeros([length, 3])

        # Left hand point is special because of evenness
        # 0 refers to the coefficient of the point, rather than left of the point
        self._dydxstencil[0, 0] = invdoublediffs[0] - invdiffs[1]
        self._dydxstencil[0, 1] = invdiffs[1] - invdoublediffs[0]
        self._dydxstencil[0, 2] = 0

        # Middle points are straightforward
        for i in range(1, length-1):
            self._dydxstencil[i, 0] = -invdiffs[i] + invdoublediffs[i]
            self._dydxstencil[i, 1] = invdiffs[i] - invdiffs[i+1]
            self._dydxstencil[i, 2] = invdiffs[i+1] - invdoublediffs[i]

        # Right hand point needs a slightly different form, still at O(h^2)
        # 2 refers to the coefficient of the point, rather than right of the point
        i = length - 1
        self._dydxstencil[i, 0] = invdiffs[i-1] - invdoublediffs[i-1]
        self._dydxstencil[i, 1] = -invdiffs[i-1] - invdiffs[i]
        self._dydxstencil[i, 2] = invdoublediffs[i-1] + invdiffs[i]


        # Construct rhoderiv. Note that as a second derivative, we can't compute this
        # for the last gridpoint.
        # Start by putting together the pieces
        self._rhostencil = np.zeros([length - 1, 3])
        x4vals = xvals**4
        x4diffs = np.insert(np.diff(x4vals), 0, 0)
        x4doublediffs = x4diffs[1:] + x4diffs[:-1]
        # x4doublediffs[i] = x[i+1]^4 - x[i-1]^4
        x4sums = np.insert(x4vals[1:] + x4vals[:-1], 0, 2*x4vals[0])
        x4sums /= diffs
        x4sums *= 4/3
        # x4sums[i] = 4/3*(x[i]^4 + x[i-1]^4) / (x[i] - x[i-1])

        # Construct the first element (backward difference vanishes)
        self._rhostencil[0, 0] = - x4sums[1]
        self._rhostencil[0, 1] = x4sums[1]
        self._rhostencil[0, 2] = 0
        self._rhostencil[0] /= x4doublediffs[0]

        # Construct the rest of the elements
        for i in range(1, length - 1):
            self._rhostencil[i, 0] = x4sums[i]
            self._rhostencil[i, 1] = - x4sums[i+1] - x4sums[i]
            self._rhostencil[i, 2] = x4sums[i+1]
            self._rhostencil[i] /= x4doublediffs[i]

    def dydx(self, yvals):
        """
        Pass in a vector of y values
        Returns a vector of dy/dx values
        """
        return self._compute_deriv(yvals, self._dydxstencil)

    def rhoderiv(self, yvals):
        """
        Pass in a vector of y values
        Returns a vector of (x d^2y/dx^2 + 4 dy/dx)/3 values
        Note that the this is not computed for the last gridpoint
        """
        return self._compute_deriv(yvals, self._rhostencil)

    def _compute_deriv(self, yvals, stencil):
        """
        Pass in a vector of y values and a stencil
        Computes the action of the stencil on the y values
        """
        if self.length != len(yvals):
            raise DerivativeError("xvals and yvals have different dimensions")

        derivatives = np.zeros(len(stencil))
        for pos in range(len(stencil)):
            derivatives[pos] = np.dot(stencil[pos], yvals[self.seqs[pos]])
        return derivatives

    def rightdydx(self, yvals):
        """
        Pass in a vector of y values
        Returns the derivative at the last position
        """
        if len(self._xvals) != len(yvals):
            raise DerivativeError("xvals and yvals have different dimensions")
        return np.dot(self._dydxstencil[-1], yvals[self.length-3:])

# Testing suite
if __name__ == "__main__":
    import random
    from math import pi

    numvals = 40

    # Randomly pick some x values
    x = np.sort(np.array([random.uniform(0.0, 2*pi) for i in range(numvals)]))
    # x = np.array([i for i in range(20)]) + 0.5
    # x /= 2

    # Take some trig functions
    ysin = np.sin(x)
    ycos = np.cos(x)

    # Initialize differentiator
    diff = Derivative(x)

    # Take derivatives
    dycos = diff.dydx(ycos)
    # How did we go?
    print("x", "Derivative", "Actual", "Error")
    for i in range(len(dycos)):
        print(x[i], dycos[i], -ysin[i], abs(dycos[i] + ysin[i]))

    # Take rho derivatives
    rho = diff.rhoderiv(ycos)
    actual = (-x*ycos-4*ysin)/3
    # How did we go?
    print("x", "Derivative", "Actual", "Error")
    for i in range(len(rho)):
        print(x[i], rho[i], actual[i], abs(rho[i] - actual[i]))

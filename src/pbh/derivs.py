"""Finite difference derivatives on a non-uniform grid with reflection symmetry about the origin.

Contains the :class:`Derivative` class, which computes second order finite difference derivatives.

* Assumes even or odd symmetry about the origin
* Assumes no gridpoint at the origin
* Computes dy/dx for even or odd functions
* Computes (x d^2y/dx^2 + 4 dy/dx)/3 for even functions in two different ways (one with a stencil, one from y(A) and
  x(A) for a Lagrangian grid)
"""

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]


class DerivativeError(Exception):
    """Class used for all errors taking derivatives."""


class Derivative:
    """Finite difference derivatives for functions on a grid without a gridpoint at the origin.

    The following derivatives are computed:

    * dy/dx
    * rhoderiv = (x d^2y/dx^2 + 4 dy/dx) / 3 = 4/3 * d/d(x^4) (x^4 dy/dx)
    """

    def __init__(self, xvals: FloatArray) -> None:
        """Initialize all of the derivative coefficients in our stencils.

        Args:
            xvals: Sorted array of (positive) x values.
        """
        # Initialize the stencil storage
        self.length = length = len(xvals)
        if length < 3:
            raise DerivativeError("Grid too short to compute derivatives")

        # Construct the xvals differences
        # diffs[i] is xvals[i] - xvals[i-1], which goes backwards so we can get
        # the boundary spacing correct
        diffs = np.insert(np.diff(xvals), 0, xvals[0] * 2)
        invdiffs = 1 / diffs
        # doublediffs[i] = x[i+1] - x[i-1]
        doublediffs = diffs[1:] + diffs[:-1]
        invdoublediffs = 1 / doublediffs

        # Make the slices that will extract the correct components for dot products
        self.seqs = [slice(i - 1, i + 2) for i in range(length)]
        self.seqs[0] = slice(0, 3)
        self.seqs[-1] = slice(length - 3, length)

        # Construct the simple linear derivative
        # This is a centered nonuniform first order derivative using three gridpoints
        # to be accurate to O(h^2)
        # df/dx = (f[i]-f[i-1])/(x[i]-x[i-1]) + (f[i+1]-f[i])/(x[i+1]-x[i])
        #           + (f[i+1]-f[i-1])/(x[i+1]-x[i-1])
        self._evenstencil = np.zeros([length, 3])

        # Left hand point is special because of evenness
        # This comes from a 4-point stencil
        # 0 index refers to the coefficient of the point, rather than left of the point
        self._evenstencil[0, 0] = +1 / (xvals[0] - xvals[1]) + 1 / (xvals[0] + xvals[1])
        self._evenstencil[0, 1] = -1 / (xvals[0] - xvals[1]) - 1 / (xvals[0] + xvals[1])
        self._evenstencil[0, 2] = 0

        # Middle points are straightforward
        for i in range(1, length - 1):
            self._evenstencil[i, 0] = -invdiffs[i] + invdoublediffs[i]
            self._evenstencil[i, 1] = invdiffs[i] - invdiffs[i + 1]
            self._evenstencil[i, 2] = invdiffs[i + 1] - invdoublediffs[i]

        # Right hand point needs a slightly different form, still at O(h^2)
        # 2 index refers to the coefficient of the point, rather than right of the point
        i = length - 1
        self._evenstencil[i, 0] = invdiffs[i - 1] - invdoublediffs[i - 1]
        self._evenstencil[i, 1] = -invdiffs[i - 1] - invdiffs[i]
        self._evenstencil[i, 2] = invdoublediffs[i - 1] + invdiffs[i]

        # Make a different version for odd derivatives
        # Only two coefficients need to change
        # Note that this comes from a 4-point derivative
        self._oddstencil = self._evenstencil.copy()
        self._oddstencil[0, 0] = 1 / xvals[0] + 1 / (xvals[0] - xvals[1]) + 1 / (xvals[0] + xvals[1])
        self._oddstencil[0, 1] = -2 / xvals[1] + 1 / (xvals[1] - xvals[0]) + 1 / (xvals[0] + xvals[1])

        # Construct rhoderiv. Note that as a second derivative, we can't compute this
        # for the last gridpoint.
        # Start by putting together the pieces
        self._rhostencil = np.zeros([length - 1, 3])
        x4vals = xvals**4
        x4diffs = np.insert(np.diff(x4vals), 0, 0)
        x4doublediffs = x4diffs[1:] + x4diffs[:-1]
        # x4doublediffs[i] = x[i+1]^4 - x[i-1]^4
        x4sums = np.insert(x4vals[1:] + x4vals[:-1], 0, 2 * x4vals[0])
        x4sums *= invdiffs * 4 / 3
        # x4sums[i] = 4/3*(x[i]^4 + x[i-1]^4) / (x[i] - x[i-1])

        # Construct the first element (uses special formula)
        h = diffs[0]
        epsilon = diffs[1] / h - 1
        self._rhostencil[0, 0] = -5 / 3 / h / (1 + epsilon) / (2 + epsilon)
        self._rhostencil[0, 1] = -self._rhostencil[0, 0]
        self._rhostencil[0, 2] = 0

        # Construct the rest of the elements
        for i in range(1, length - 1):
            self._rhostencil[i, 0] = x4sums[i]
            self._rhostencil[i, 1] = -x4sums[i + 1] - x4sums[i]
            self._rhostencil[i, 2] = x4sums[i + 1]
            self._rhostencil[i] /= x4doublediffs[i]

    def dydx(self, yvals: FloatArray, even: bool) -> FloatArray:
        """Compute dy/dx.

        Args:
            yvals: Values of y on the grid.
            even: Parity of y about the origin (True for even, False for odd).

        Returns:
            A vector of dy/dx values.
        """
        stencil = self._evenstencil if even else self._oddstencil
        return self._compute_deriv(yvals, stencil)

    def rhoderiv(self, yvals: FloatArray) -> FloatArray:
        """Compute (x d^2y/dx^2 + 4 dy/dx)/3 for an even function y.

        Note that this is not computed for the last gridpoint; instead, 0 is returned there.
        """
        return self._compute_deriv(yvals, self._rhostencil)

    @staticmethod
    def rhoderiv_lagrange(yvals: FloatArray, rvals: FloatArray) -> FloatArray:
        """Compute (r d^2y/dr^2 + 4 dy/dr)/3 where y and r are both given as functions of a grid coordinate.

        Because this depends on both yvals and rvals, it cannot be computed with a fixed stencil.
        Note that this is not computed for the last gridpoint; instead, 0 is returned there.
        """
        # Start by computing the differences in yvals
        ydiffs = np.diff(yvals)
        rdiffs = np.diff(rvals)

        # Compute dy/dr as a forwards difference
        dydr = ydiffs / rdiffs

        # Compute the r^4 terms
        r4vals = rvals**4
        r4sums = r4vals[1:] + r4vals[:-1]
        r4diffs = np.diff(r4vals)
        r4doublediffs = r4diffs[1:] + r4diffs[:-1]

        # Construct the result
        result = 4 / 3 * np.diff(r4sums * dydr) / r4doublediffs
        # Set first and last gridpoints to zero
        fullresult = np.zeros_like(yvals)
        # Insert the results
        fullresult[1:-1] = result

        # Need something special for the first gridpoint (assuming evenness)
        fullresult[0] = 10 / 3 * ydiffs[0] * rvals[0] / rdiffs[0] / (rvals[0] + rvals[1])

        # Return the result
        return fullresult

    def _compute_deriv(self, yvals: FloatArray, stencil: FloatArray) -> FloatArray:
        """Compute the action of a stencil on the y values.

        Stencils with fewer rows than gridpoints leave the trailing entries of the result as zero.
        """
        if self.length != len(yvals):
            raise DerivativeError("xvals and yvals have different dimensions")

        derivatives = np.zeros(len(yvals))
        # This is unfortunately slow in python, but I can't figure out a way to get numpy to do it more quickly!
        for pos in range(len(stencil)):
            derivatives[pos] = np.dot(stencil[pos], yvals[self.seqs[pos]])
        return derivatives

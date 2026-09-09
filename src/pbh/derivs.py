"""Finite difference derivatives on a non-uniform grid with reflection symmetry about the origin.

Contains the :class:`Derivative` class, which computes second order finite difference derivatives.

* Assumes even or odd symmetry about the origin
* Assumes no gridpoint at the origin
* Computes dy/dx for even or odd functions
* Computes (x d^2y/dx^2 + 4 dy/dx)/3 for even functions in two different ways (one with a stencil, one from y(A) and
  x(A) for a Lagrangian grid)
"""

import numpy as np
import scipy.sparse as sp
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

        # Each stencil has three coefficients per gridpoint, applying to the three columns starting at
        # starts[i]: the point and its two neighbours in the interior, and the three edge points at either end.
        starts = np.arange(-1, length - 1)
        starts[0] = 0
        starts[-1] = length - 3

        # Construct the simple linear derivative
        # This is a centered nonuniform first order derivative using three gridpoints
        # to be accurate to O(h^2)
        # df/dx = (f[i]-f[i-1])/(x[i]-x[i-1]) + (f[i+1]-f[i])/(x[i+1]-x[i])
        #           + (f[i+1]-f[i-1])/(x[i+1]-x[i-1])
        evenstencil = np.zeros([length, 3])

        # Left hand point is special because of evenness
        # This comes from a 4-point stencil
        # 0 index refers to the coefficient of the point, rather than left of the point
        evenstencil[0, 0] = +1 / (xvals[0] - xvals[1]) + 1 / (xvals[0] + xvals[1])
        evenstencil[0, 1] = -1 / (xvals[0] - xvals[1]) - 1 / (xvals[0] + xvals[1])
        evenstencil[0, 2] = 0

        # Middle points are straightforward (row i uses diffs[i], diffs[i+1] and doublediffs[i])
        evenstencil[1:-1, 0] = -invdiffs[1:-1] + invdoublediffs[1:]
        evenstencil[1:-1, 1] = invdiffs[1:-1] - invdiffs[2:]
        evenstencil[1:-1, 2] = invdiffs[2:] - invdoublediffs[1:]

        # Right hand point needs a slightly different form, still at O(h^2)
        # 2 index refers to the coefficient of the point, rather than right of the point
        i = length - 1
        evenstencil[i, 0] = invdiffs[i - 1] - invdoublediffs[i - 1]
        evenstencil[i, 1] = -invdiffs[i - 1] - invdiffs[i]
        evenstencil[i, 2] = invdoublediffs[i - 1] + invdiffs[i]

        # Make a different version for odd derivatives
        # Only two coefficients need to change
        # Note that this comes from a 4-point derivative
        oddstencil = evenstencil.copy()
        oddstencil[0, 0] = 1 / xvals[0] + 1 / (xvals[0] - xvals[1]) + 1 / (xvals[0] + xvals[1])
        oddstencil[0, 1] = -2 / xvals[1] + 1 / (xvals[1] - xvals[0]) + 1 / (xvals[0] + xvals[1])

        # Construct rhoderiv. Note that as a second derivative, we can't compute this
        # for the last gridpoint.
        # Start by putting together the pieces
        rhostencil = np.zeros([length - 1, 3])
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
        rhostencil[0, 0] = -5 / 3 / h / (1 + epsilon) / (2 + epsilon)
        rhostencil[0, 1] = -rhostencil[0, 0]
        rhostencil[0, 2] = 0

        # Construct the rest of the elements (row i uses x4sums[i], x4sums[i+1] and x4doublediffs[i])
        rhostencil[1:, 0] = x4sums[1:-1]
        rhostencil[1:, 1] = -x4sums[2:] - x4sums[1:-1]
        rhostencil[1:, 2] = x4sums[2:]
        rhostencil[1:] /= x4doublediffs[1:, np.newaxis]

        # Assemble the stencils into sparse operators
        self._even_operator = self._make_operator(evenstencil, starts)
        self._odd_operator = self._make_operator(oddstencil, starts)
        self._rho_operator = self._make_operator(rhostencil, starts)

    def _make_operator(self, stencil: FloatArray, starts: NDArray[np.intp]) -> sp.csr_array:
        """Assemble a (rows, 3) stencil into a sparse (length, length) matrix.

        Row i of the stencil applies to the three consecutive columns beginning at starts[i]. Stencils with fewer rows
        than gridpoints leave the trailing rows of the operator empty.
        """
        rows = len(stencil)
        row_index = np.repeat(np.arange(rows), 3)
        col_index = (starts[:rows, np.newaxis] + np.arange(3)).ravel()
        matrix = sp.csr_array((stencil.ravel(), (row_index, col_index)), shape=(self.length, self.length))
        matrix.sort_indices()
        return matrix

    def dydx(self, yvals: FloatArray, even: bool) -> FloatArray:
        """Compute dy/dx.

        Args:
            yvals: Values of y on the grid.
            even: Parity of y about the origin (True for even, False for odd).

        Returns:
            A vector of dy/dx values.
        """
        return self._apply(self._even_operator if even else self._odd_operator, yvals)

    def rhoderiv(self, yvals: FloatArray) -> FloatArray:
        """Compute (x d^2y/dx^2 + 4 dy/dx)/3 for an even function y.

        Note that this is not computed for the last gridpoint; instead, 0 is returned there.
        """
        return self._apply(self._rho_operator, yvals)

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

    def _apply(self, operator: sp.csr_array, yvals: FloatArray) -> FloatArray:
        """Apply a sparse derivative operator to the y values."""
        if self.length != len(yvals):
            raise DerivativeError("xvals and yvals have different dimensions")
        return operator @ yvals

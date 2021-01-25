# -*- coding: utf-8 -*-
"""
Computes finite difference derivatives on a non-uniform grid
"""
import numpy as np
np.seterr(all='raise')

class DerivativeError(Exception):
    pass

class Derivative(object):
    """
    Computes a finite difference derivative on a non-uniform grid at a given order
    * Use set_x to pass in a set of x values and construct the appropriate stencil
    * Then use dydx to compute derivatives at those x values

    Even/Odd boundary conditions can be specified when constructing
    the stencil initially. Alternatively, a given stencil can be modified to
    change the type of boundary condition later (eg, from even to odd)
    """

    def __init__(self, order):
        """
        Sets the order of the finite difference code
        """
        # Stores the number of grid points in a stencil
        self.N = order + 1
        # Store the number of points to the left to use in the derivative
        # If odd, uses more points to the right by default
        self.leftpoints = order // 2
        # Initialize a dummy stencil
        self.stencil = np.array([[0.0]])

    def set_x(self, xvals, boundary=0):
        """
        Pass in an array of x values
        Computes the weights of a derivative operation for a given set of y
        Allows for different boundary conditions at x=0:
            -1 = odd
            0 = no boundary
            1 = even
        Even and odd boundary conditions assume that the first data point is
        at x > 0
        """
        length = len(xvals)
        if length < self.N:
            raise DerivativeError("Grid too short for given order")

        # Initialize the stencil
        if np.shape(self.stencil) == (length, length):
            self.stencil.fill(0.0)
        else:
            # Make a new array
            self.stencil = np.zeros([length, length])

        # Do gridpoints on left side first
        lp = 0
        rp = self.N
        for i in range(self.leftpoints):
            # Which boundary condition will we use?
            if boundary == 0:
                # No boundary condition
                self._default_weights(i - lp, xvals[lp:rp], self.stencil[i, lp:rp])
            else:
                # Even/Odd
                self._boundary_weights(i - lp, xvals[lp:rp], self.stencil[i, lp:rp], boundary)

        # Do gridpoints in middle next
        for i in range(self.leftpoints, length - self.N + self.leftpoints + 1):
            # Find the left- and right-most points in the stencil
            # Note that rp is the index of the rightmost point + 1
            lp = i - self.leftpoints
            rp = lp + self.N
            self._default_weights(i - lp, xvals[lp:rp], self.stencil[i, lp:rp])

        # Do gridpoints on right side last
        for i in range(length - self.N + self.leftpoints + 1, length):
            lp = length - self.N
            rp = length
            self._default_weights(i - lp, xvals[lp:rp], self.stencil[i, lp:rp])

    def apply_boundary(self, xvals, boundary=0):
        """
        Applies a boundary condition to the stencil for the left hand points.
        Same arguments as set_x
        Can be used to change boundary conditions without recomputing the entire
        stencil (eg, from even to odd).
        Make sure that the stencil has already been set for the given xvals,
        else only the leftmost points will differentiate correctly.
        """
        # Check that the stencil is the correct shape
        length = len(xvals)
        if np.shape(self.stencil) != (length, length):
            raise DerivativeError("Stencil is wrong shape for these x values")

        # Go and update the stencils for the points on the left
        lp = 0
        rp = self.N
        for i in range(self.leftpoints):
            # Which boundary condition will we use?
            if boundary == 0:
                # No boundary condition
                self._default_weights(i - lp, xvals[lp:rp], self.stencil[i, lp:rp])
            else:
                # Even/Odd
                self._boundary_weights(i - lp, xvals[lp:rp], self.stencil[i, lp:rp], boundary)

    def _default_weights(self, i, xvals, stencil):
        """
        Compute the weights for a given grid point
        No fancy boundary conditions here
        i is the point of interest
        xvals is the slice of data
        stencil is where we will save the resulting weights

        Both xvals and stencil are vectors of length N
        """

        for j in range(self.N):
            # We do the i case by adding together all the rest of the results
            if j == i:
                continue
            # We compute l'_j(x_i)
            # Start with the denominator
            denom = 1.0
            xdiff = xvals[j] - xvals
            for a in range(self.N):
                if a == j:
                    continue
                denom *= xdiff[a]
            # Now do the numerator
            num = 1.0
            xdiff = xvals[i] - xvals
            for b in range(self.N):
                if b == j or b == i:
                    continue
                num *= xdiff[b]
            # Compute the weight
            stencil[j] = num / denom

        # Add the contribution to the ith component
        stencil[i] = 0.0
        stencil[i] = - np.sum(stencil)

    def _boundary_weights(self, i, xvals, stencil, multiplier):
        """
        Compute the weights for a given grid point using even/odd boundary
        conditions at the origin.
        i is the point of interest
        xvals is the slice of data
        stencil is where we will save the resulting weights
        multiplier = +1 for even, -1 for odd

        Both xvals and stencil are vectors of length N
        """

        # How many points are off the end with an even/odd boundary?
        delta = self.leftpoints - i
        # Construct a new xvals vector from this data
        newxvals = np.concatenate((-xvals[delta - 1::-1], xvals[0:self.N-delta]))
        # Construct a new i value for newxvals
        newi = i + delta
        # Make a new stencil placeholder
        newstencil = np.zeros_like(stencil)

        # Construct the stencil for newi, newxvals, store in newstencil
        self._default_weights(newi, newxvals, newstencil)

        # If the boundary condition is odd, flip the appropriate signs
        if multiplier == -1:
            for i in range(delta):
                newstencil[i] *= -1

        # Reconstruct the stencil for the original xvals
        # Start by clearing it (important for when recomputing stencils)
        stencil.fill(0.0)
        # Copy over the unflipped components
        for i in range(self.N - delta):
            stencil[i] = newstencil[i + delta]
        # Now add in the flipped components
        for i in range(delta):
            stencil[delta - i - 1] += newstencil[i]

    def dydx(self, yvals):
        """
        Pass in a vector of y values
        Returns a vector of dy/dx values
        Must use set_x to construct the stencil first
        """
        return np.dot(self.stencil, yvals)

    def leftdydx(self, yvals):
        """
        Pass in a vector of y values
        Returns the derivative at the first position
        Must use set_x to construct the stencil first
        """
        return np.dot(self.stencil[0], yvals)

    def rightdydx(self, yvals):
        """
        Pass in a vector of y values
        Returns the derivative at the last position
        Must use set_x to construct the stencil first
        """
        return np.dot(self.stencil[-1], yvals)

# Testing suite
if __name__ == "__main__":
    import random
    from math import pi

    numvals = 40

    # Randomly pick some x values
    x = np.sort(np.array([random.uniform(0.0, 2*pi) for i in range(numvals)]))

    # Take some trig functions
    ysin = np.sin(x)
    ycos = np.cos(x)

    # Initialize differentiator
    diff = Derivative(4)
    diff.set_x(x)

    # Take derivatives with no boundary conditions
    dycos = diff.dydx(ycos)
    dysin = diff.dydx(ysin)

    # Take derivatives with boundary conditions
    diff.apply_boundary(x, 1)
    dycos2 = diff.dydx(ycos)
    diff.apply_boundary(x, -1)
    dysin2 = diff.dydx(ysin)

    # How did we go?
    print("x", "Actual", "No boundary", "Boundary", "Error 1", "Error2")
    for i in range(numvals):
        print(x[i], ycos[i], dysin[i], dysin2[i], dysin[i] - ycos[i], dysin2[i] - ycos[i])
        print(x[i], -ysin[i], dycos[i], dycos2[i], dycos[i] + ysin[i], dycos2[i] + ysin[i])

"""The maps that place the faces in the scaled areal radius (paper Section 7.1; the pinned map of Section 8.1).

The grid has `N` cells and faces `j = 0..N`: face `0` is the origin, face `N` the outer boundary at the fixed scaled
areal radius `Rtilde_max` (the paper's `Rtilde_max` of Sections 5 and 8, `X_N` in the stencils), and one virtual
face `N + 1` beyond it serves the virtual cell of `geometry.py`. Where the faces sit is the map: a rule for the
radius `X_j` of face `j` at time `xi`, evaluated analytically together with the face's velocity `(d_xi X)_j`, and
never evolved, so that radii and volumes carry no truncation error and no constraint of their own.

The paper writes the map on a continuous label `x` with faces at `x_j = j h`, `h = x_max / N`, and calls `X(xi, x)`
the map. Here the label is the fraction `u = j / N` of the way to the outer face and the outer radius is a parameter
of the map: the same maps with `x_max = 1`, and one coordinate fewer to keep in mind (owner, 2026-09-20). Nothing
in the scheme ever differences in the label (Section 7.1), so the Jacobian `d_x X` is not needed; the Courant step
uses the cell widths in the radius.

A map is admissible if `X_0 = 0` exactly and the radii increase strictly. This module holds the maps of Section 7:
the identity, useful in tests because it is uniform in the radius; a static stretch, the production choice, which
refines the grid near the origin, where everything that needs resolution happens, at no cost in the scheme; and the
pinned map, a static map with every face held at fixed physical radius from a pin-on time, which is the inner half
of the post-formation map of Section 8.1 and on its own the test map of the deviation form (Section 7.6). The blend
that joins the pinned map to its static base is added with the post-formation bites.

Every map is a frozen dataclass deriving from the abstract `Map`, with one method, `radii(xi, N)`, returning
`(X, X_xi)` at the `N + 2` faces, and one property, `is_static`, saying whether the geometry may be computed once and
cached (Section 7.1).
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from pbh.types import FloatArray

type MapValues = tuple[FloatArray, FloatArray]
"""`(X, X_xi)`: the radii of the faces `0..N+1` and their velocities at fixed face number."""


def fractions(N: int) -> FloatArray:
    """The label `u_j = j / N` of the faces `0..N+1`, the last being the virtual face at `(N + 1) / N`."""
    if N < 2:
        raise ValueError(f"need N >= 2 cells, got N = {N}")
    return np.arange(N + 2, dtype=np.float64) / N


class Map(ABC):
    """What every map provides: the face radii at a time, and whether they depend on it."""

    @property
    @abstractmethod
    def is_static(self) -> bool:
        """True if the radii do not depend on `xi`, so that the geometry can be computed once and cached."""

    @abstractmethod
    def radii(self, xi: float, N: int) -> MapValues:
        """`X_j` and `(d_xi X)_j` at the faces `0..N+1` at time `xi`, for a grid of `N` cells."""


@dataclass(frozen=True)
class IdentityMap(Map):
    """Faces uniform in the radius, `X_j = Rtilde_max j / N` (the paper's `X = x`).

    Useful in tests, where a uniform grid in the radius keeps the arithmetic transparent; the background is at rest
    on it. Production runs use a stretch.

    Attributes:
        Rtilde_max: The scaled areal radius of the outer face.
    """

    Rtilde_max: float

    def __post_init__(self) -> None:
        """The outer radius must be positive."""
        if self.Rtilde_max <= 0.0:
            raise ValueError(f"Rtilde_max must be positive, got {self.Rtilde_max}")

    @property
    def is_static(self) -> bool:
        """Static: the radii do not depend on `xi`."""
        return True

    def radii(self, xi: float, N: int) -> MapValues:
        """`X_j = Rtilde_max j / N`, `d_xi X = 0`."""
        X = self.Rtilde_max * fractions(N)
        return X, np.zeros_like(X)


@dataclass(frozen=True)
class SinhStretch(Map):
    """`X = L sinh(u asinh(Rtilde_max / L))`, `u = j / N`: a static stretch that concentrates cells near the origin.

    The radius grows like `u` near the origin and exponentially far out, so the cells are finest at the centre and
    coarsen smoothly outward, with `X_N = Rtilde_max` exactly and no change to any stencil: the scheme contains
    nothing about the spacing of the faces except through the geometry. The paper verifies FRW on the stretch
    `X = 3 sinh(x / 3)` (Section 7.3), which is this map with `L = 3`. Odd in `u`, `X_0 = 0` exactly, strictly
    increasing. This is the production base map.

    Attributes:
        Rtilde_max: The scaled areal radius of the outer face.
        scale: The length `L` in units of the scaled areal radius: the grid is nearly uniform inside `L` and
            stretches beyond it; the smaller `L`, the more of the cells sit near the origin.
    """

    Rtilde_max: float
    scale: float

    def __post_init__(self) -> None:
        """Both lengths must be positive for the map to be admissible."""
        if self.Rtilde_max <= 0.0 or self.scale <= 0.0:
            raise ValueError(f"Rtilde_max and scale must be positive, got {self.Rtilde_max}, {self.scale}")

    @property
    def is_static(self) -> bool:
        """Static: the radii do not depend on `xi`."""
        return True

    def radii(self, xi: float, N: int) -> MapValues:
        """`X = L sinh(u asinh(Rtilde_max / L))`, `d_xi X = 0`."""
        X = self.scale * np.sinh(fractions(N) * math.asinh(self.Rtilde_max / self.scale))
        return X, np.zeros_like(X)


@dataclass(frozen=True)
class PinnedMap(Map):
    """`X = e^(-alpha (xi - xi_on)) B`: a static map `B`, every face pinned to a fixed physical radius from `xi_on`.

    The physical areal radius is `Rbar = e^(alpha xi) X` in units of `R_H`, so on this map `Rbar = e^(alpha xi_on)
    B_j` at every time: from the pin-on time `xi_on` the faces stand still in physical radius while the background
    expands through them, contracting in the scaled coordinate at `d_xi X = -alpha X`. At `xi_on` the map coincides
    with its base, so a state on the base carries over unchanged when the pin is switched on. That is what the grid
    must do next to a black hole, whose radius is fixed in physical units while a comoving grid would run away from
    it, and the post-formation map of Section 8.1 is exactly this map inside its transition and the static `B`
    outside, joined by a smooth step (with the ramp `T` of eq:numbh:map in place of `xi - xi_on`).

    On its own it is a test map only, because it moves the outer face, which Section 8.1 forbids: the outer rows of
    Section 7.5 are written for `(d_xi X)_N = 0`, and the energy estimate of Section 7.4 fails at a moving outer
    face at a rate proportional to the number of cells. Its test value is that it moves everywhere: Section 7.6 uses
    it to show why the integrator works in deviation form (`1e-15` against `1e-8` for the direct form over one
    e-fold), and the volume rate on it is known exactly (`d_xi Delta V = -3 alpha Delta V`).

    Attributes:
        base: The static map `B` being pinned, an `IdentityMap` or a `SinhStretch`.
        alpha: The scale-factor exponent `alpha = 2 / (3 (1 + w))` of the fluid.
        xi_on: The time from which the faces are pinned; the map is its base there.
    """

    base: Map
    alpha: float
    xi_on: float = 0.0

    def __post_init__(self) -> None:
        """Only a static map can be pinned; pinning a moving one would compound two motions."""
        if not self.base.is_static:
            raise ValueError("the base of a pinned map must be static")

    @property
    def is_static(self) -> bool:
        """Moving: the radii depend on `xi`, so the geometry is recomputed at every stage."""
        return False

    def radii(self, xi: float, N: int) -> MapValues:
        """`X = e^(-alpha (xi - xi_on)) B`, `d_xi X = -alpha X`."""
        B, _ = self.base.radii(0.0, N)  # the base is static, so its time argument is immaterial
        X = math.exp(-self.alpha * (xi - self.xi_on)) * B
        return X, -self.alpha * X

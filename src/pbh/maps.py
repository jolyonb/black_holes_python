"""The label grid and the map from labels to the scaled areal radius (paper Section 7.1).

The grid is uniform in the label `x`: faces at `x_j = j h`, `j = 0..N`, `h = x_max / N`, plus one virtual label
`x_{N+1} = x_max + h` for the virtual cell of `geometry.py`. Where the faces sit in the scaled areal radius
`X = Rtilde` is the map `X(xi, x)`, which is evaluated at the faces analytically, together with its two derivatives
`(d_xi X)` (how fast a face moves, at fixed label) and `(d_x X)` (the Jacobian), and never evolved: radii and volumes
then carry no truncation error and no constraint of their own.

A map is admissible if `X(xi, 0) = 0` exactly and `d_x X > 0` (Section 7.1). This module holds the maps of
Section 7: the identity, useful in tests because on it the label is the radius; a static stretch, the production choice,
which refines the grid near the origin, where everything that needs resolution happens, at no cost in the scheme,
since the stencils contain neither `h` nor `d_x X`; and the pinned map, a static map with every face held at fixed
physical radius, which is the inner half of the post-formation map of Section 8.1 and on its own the test map of the
deviation form (Section 7.6). The blend that joins the pinned map to its static base will be added later.

Every map is a frozen dataclass deriving from the abstract `Map`, with one method, `at(xi, x)`, returning the
triple `(X, X_xi, X_x)` at the given labels, and one property, `is_static`, saying whether the geometry may be
computed once and cached (Section 7.1).
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from pbh.types import FloatArray

type MapValues = tuple[FloatArray, FloatArray, FloatArray]
"""`(X, X_xi, X_x)` at the labels a map was evaluated at."""


def face_labels(N: int, x_max: float) -> FloatArray:
    """The `N + 2` labels `x_0 .. x_{N+1}` of the uniform grid: the `N + 1` faces plus the virtual face beyond.

    Args:
        N: The number of cells.
        x_max: The label of the outer face.

    Returns:
        `j h` for `j = 0..N+1` with `h = x_max / N`; `x_0 = 0` exactly and `x_N = x_max`.
    """
    if N < 2 or x_max <= 0.0:
        raise ValueError(f"need N >= 2 cells and x_max > 0, got N = {N}, x_max = {x_max}")
    return np.arange(N + 2, dtype=np.float64) * (x_max / N)


class Map(ABC):
    """What every map provides: its values at given labels, and whether it depends on `xi`."""

    @property
    @abstractmethod
    def is_static(self) -> bool:
        """True if `X` does not depend on `xi`, so that the geometry can be computed once and cached."""

    @abstractmethod
    def at(self, xi: float, x: FloatArray) -> MapValues:
        """Evaluate `X`, `(d_xi X)` and `(d_x X)` at time `xi` and the labels `x`."""


@dataclass(frozen=True)
class IdentityMap(Map):
    """`X = x`: the comoving-scaled areal radius is the label itself (Section 7.1).

    Useful in tests, where a uniform grid in the radius keeps the arithmetic transparent; the background is at rest on
    it. Production runs use a stretch.
    """

    @property
    def is_static(self) -> bool:
        """Static: `X` does not depend on `xi`."""
        return True

    def at(self, xi: float, x: FloatArray) -> MapValues:
        """`X = x`, `d_xi X = 0`, `d_x X = 1`."""
        return x.copy(), np.zeros_like(x), np.ones_like(x)


@dataclass(frozen=True)
class SinhStretch(Map):
    """`X = L sinh(x / L)`: a static stretch that concentrates cells near the origin (Section 7.1).

    The Jacobian `cosh(x / L)` is `1` at the origin and grows outward, so in areal radius the cells are finest at the
    centre and coarsen smoothly, with no change to any stencil: the scheme contains neither `h` nor `d_x X` except in
    the Courant number. Odd in `x`, `X(0) = 0` exactly, `d_x X >= 1 > 0`.

    Attributes:
        scale: The length `L` in label units; the map is the identity to relative order `(x / L)^2`.
    """

    scale: float

    def __post_init__(self) -> None:
        """The scale must be positive for the map to be admissible."""
        if self.scale <= 0.0:
            raise ValueError(f"the stretch scale must be positive, got {self.scale}")

    @property
    def is_static(self) -> bool:
        """Static: `X` does not depend on `xi`."""
        return True

    def at(self, xi: float, x: FloatArray) -> MapValues:
        """`X = L sinh(x / L)`, `d_xi X = 0`, `d_x X = cosh(x / L)`."""
        return self.scale * np.sinh(x / self.scale), np.zeros_like(x), np.cosh(x / self.scale)


@dataclass(frozen=True)
class PinnedMap(Map):
    """`X = e^(-alpha xi) B(x)`: a static map `B` with every face pinned to a fixed physical radius (test map).

    The physical areal radius is `Rbar = e^(alpha xi) X` in units of `R_H`, so on this map `Rbar = B(x)` at every
    time: the faces stand still in physical radius while the background expands through them, contracting in the
    scaled coordinate at `d_xi X = -alpha X`. That is what the grid must do next to a black hole, whose radius is fixed
    in physical units while a comoving grid would run away from it, and the post-formation map of Section 8.1 is
    exactly this map inside its transition and the static `B` outside, joined by a smooth step.

    On its own it is a test map only, because it moves the outer face, which Section 8.1 forbids: the outer rows of
    Section 7.5 are written for `(d_xi X)_N = 0`, and the energy estimate of Section 7.4 fails at a moving outer face
    at a rate proportional to `1 / h`. Its test value is that it moves everywhere: Section 7.6 uses it to show why the
    integrator works in deviation form (`1e-15` against `1e-8` for the direct form over one e-fold), and the volume
    rate on it is known exactly (`d_xi Delta V = -3 alpha Delta V`).

    Attributes:
        base: The static map `B` being pinned, `IdentityMap()` or a `SinhStretch`.
        alpha: The scale-factor exponent `alpha = 2 / (3 (1 + w))` of the fluid.
    """

    base: Map
    alpha: float

    def __post_init__(self) -> None:
        """Only a static map can be pinned; pinning a moving one would compound two motions."""
        if not self.base.is_static:
            raise ValueError("the base of a pinned map must be static")

    @property
    def is_static(self) -> bool:
        """Moving: `X` depends on `xi`, so the geometry is recomputed at every stage."""
        return False

    def at(self, xi: float, x: FloatArray) -> MapValues:
        """`X = e^(-alpha xi) B(x)`, `d_xi X = -alpha X`, `d_x X = e^(-alpha xi) B'(x)`."""
        B, _, B_x = self.base.at(0.0, x)  # the base is static, so its time argument is immaterial
        contraction = math.exp(-self.alpha * xi)
        X = contraction * B
        return X, -self.alpha * X, contraction * B_x

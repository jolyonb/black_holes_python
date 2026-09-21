"""The maps that place the faces in the scaled areal radius (paper Section 7.1; the maps of Section 8.1).

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
of the post-formation map of Section 8.1 and on its own the test map of the deviation form (Section 7.6); and the
blend map of Section 8.1 itself, the base pinned in zones joined by flat steps and switched on by ramps, which is
the production map after formation.

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

    @abstractmethod
    def radius_at(self, xi: float, u: FloatArray) -> FloatArray:
        """`X(xi, u)` at any label `u`, the fraction `j / N` continued off the faces to a non-integer face number.

        The horizon finder interpolates the crossing of the trapping function to `j + fraction` and asks the map for
        the radius there, which is second order on any grid; interpolating the radius between the faces is not.
        """


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

    def radius_at(self, xi: float, u: FloatArray) -> FloatArray:
        """`X = Rtilde_max u`."""
        return self.Rtilde_max * u


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
        return self.radius_at(xi, fractions(N)), np.zeros(N + 2)

    def radius_at(self, xi: float, u: FloatArray) -> FloatArray:
        """`X = L sinh(u asinh(Rtilde_max / L))`."""
        return self.scale * np.sinh(u * math.asinh(self.Rtilde_max / self.scale))


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

    def radius_at(self, xi: float, u: FloatArray) -> FloatArray:
        """`X = e^(-alpha (xi - xi_on)) B(u)`."""
        return math.exp(-self.alpha * (xi - self.xi_on)) * self.base.radius_at(0.0, u)


# --- the post-formation map: the base pinned in zones, joined by flat steps, switched on by ramps (Section 8.1) ---


def quintic_step(zeta: FloatArray) -> FloatArray:
    """The step `sigma_5` of eq:numbh:map: `0` below `-1`, `1` above `1`, a quintic between, flat at both ends.

    Between them `1/2 + (15 zeta - 10 zeta^3 + 3 zeta^5) / 16`, which is `C^2` at both ends. A `tanh` never reaches
    `0` or `1` exactly, so a map built on it is neither odd at the origin nor exactly static at the outer face; the
    quintic is the simplest step that is.
    """
    z = np.clip(zeta, -1.0, 1.0)
    return 0.5 + (15.0 * z - 10.0 * z**3 + 3.0 * z**5) / 16.0


def ramp(xi: float, xi_on: float, tau_on: float) -> tuple[float, float]:
    """The switch-on ramp `T` of eq:numbh:map and its rate: `(xi - xi_on) - tau_on (1 - e^(-(xi - xi_on) / tau_on))`.

    `T = dT/dxi = 0` up to `xi_on`, so the map agrees with the pre-formation one there in value and velocity and the
    state carries over unchanged; `0 <= dT/dxi < 1` always, which is what keeps every coordinate line from moving
    inward in physical radius; and `T -> xi - xi_on - tau_on` once the ramp is over, full pinning.
    """
    s = xi - xi_on
    if s <= 0.0:
        return 0.0, 0.0
    decay = math.exp(-s / tau_on)
    return s - tau_on * (1.0 - decay), 1.0 - decay


@dataclass(frozen=True)
class Zone:
    """One pinned zone of the blend map: the labels inside its transition, pinned from its switch-on.

    Attributes:
        xi_on: The switch-on time; the zone's ramp starts there, and the driver puts it on a step boundary.
        tau_on: The ramp time (Table tab:numbh:params: `0.3`, admissible `0.2` to `0.5`).
        x_t: The label at the centre of the transition, `c_t x_AH` at switch-on, a fraction of the unit interval.
        Delta_t: The transition's half-width, `c_Delta x_AH` at switch-on.
    """

    xi_on: float
    tau_on: float
    x_t: float
    Delta_t: float

    def __post_init__(self) -> None:
        if self.tau_on <= 0.0 or self.Delta_t <= 0.0:
            raise ValueError(f"a zone needs tau_on > 0 and Delta_t > 0, got {self.tau_on}, {self.Delta_t}")
        if self.x_t - self.Delta_t <= 0.0:
            raise ValueError(f"the transition must lie outside the origin: x_t - Delta_t = {self.x_t - self.Delta_t}")
        if self.x_t + self.Delta_t >= 1.0:
            raise ValueError(f"the outer face must be static: x_t + Delta_t = {self.x_t + self.Delta_t} is not below 1")

    @property
    def inner_edge(self) -> float:
        """The label `x_t - Delta_t` inside which the step is exactly zero."""
        return self.x_t - self.Delta_t

    @property
    def outer_edge(self) -> float:
        """The label `x_t + Delta_t` outside which the step is exactly one."""
        return self.x_t + self.Delta_t

    def step(self, u: FloatArray) -> FloatArray:
        """`chi(x) = sigma_5((x - x_t) / Delta_t)` at the labels `u`."""
        return quintic_step((u - self.x_t) / self.Delta_t)


@dataclass(frozen=True)
class BlendMap(Map):
    """The post-formation map of eq:numbh:map: the base pinned inside a transition, itself outside, ramped on.

    The base is pinned to physical radius inside the transition and left as it is outside, the two joined by the
    quintic step, with the pinning switched on through a ramp in time. With one zone this is the printed map,

        X = B(x) [chi + (1 - chi) e^(-alpha T)],   d_xi X = -alpha dT/dxi B(x) (1 - chi) e^(-alpha T),

    and the four identities the scheme uses hold: inside the transition `X = e^(-alpha T) B`, pinned up to the ramp
    and odd if `B` is; outside it `X = B` and `d_xi X = 0`, so the far zone is the pre-formation map and the outer
    face is static; the radii increase strictly; and `alpha X + d_xi X >= 0`, since `dT/dxi < 1`.

    Several zones are the same map generalised for a collapse that forms more than one horizon: a later switch-on
    pins a larger region without disturbing the pinning of the smaller one inside it. Two pins on one label would
    add their ramps and move the coordinate inward in physical radius, which the third identity forbids, so instead
    the zones partition the labels, each carrying its own ramp,

        X = B(x) sum_k w_k(x) e^(-alpha T_k),   w_0 = 1 - chi_1,  w_k = chi_k - chi_(k+1),  w_K = chi_K (static),

    the steps nested and non-overlapping from the origin outward. The identities carry over: each zone is pinned
    from its own switch-on, the outermost weight is static, the radii increase because an outer zone is pinned no
    harder than an inner one, and at a switch-on the new zone's ramp is zero so the map coincides with the previous
    one in value and velocity, which is the switch-on identity again.

    The map moves, so the geometry is recomputed at every stage, and the integrator's deviation form is what keeps
    the far zone FRW to round-off through the ramp (Section 8.1).

    Attributes:
        base: The static pre-formation map `B`.
        alpha: The scale-factor exponent of the fluid.
        zones: The pinned zones from the origin outward, each switched on no earlier than the one inside it.
    """

    base: Map
    alpha: float
    zones: tuple[Zone, ...]

    def __post_init__(self) -> None:
        if not self.base.is_static:
            raise ValueError("the base of a blend map must be static")
        if not self.zones:
            raise ValueError("a blend map needs at least one zone")
        for inner, outer in zip(self.zones, self.zones[1:], strict=False):
            if outer.inner_edge < inner.outer_edge:
                raise ValueError("the zones' transitions must not overlap, from the origin outward")
            if outer.xi_on < inner.xi_on:
                raise ValueError("an outer zone cannot be switched on before the one inside it")

    @property
    def is_static(self) -> bool:
        """Moving: the radii depend on `xi` from the first switch-on."""
        return False

    def with_zone(self, zone: Zone) -> BlendMap:
        """The map with a further zone outside the existing ones: a repeated switch-on."""
        return BlendMap(self.base, self.alpha, (*self.zones, zone))

    def weights(self, u: FloatArray) -> FloatArray:
        """The partition of unity `w_k(u)` at the labels `u`: one row per zone, a last row for the static exterior."""
        steps = [zone.step(u) for zone in self.zones]
        rows = [1.0 - steps[0]]
        rows += [steps[k] - steps[k + 1] for k in range(len(steps) - 1)]
        rows.append(steps[-1])
        return np.array(rows)

    def _factors(self, xi: float, u: FloatArray) -> tuple[FloatArray, FloatArray]:
        """`sum_k w_k e^(-alpha T_k)` and `sum_k w_k dT_k/dxi e^(-alpha T_k)` at the labels `u`."""
        w = self.weights(u)
        factor = np.zeros_like(u)
        rate = np.zeros_like(u)
        for k, zone in enumerate(self.zones):
            T, dT = ramp(xi, zone.xi_on, zone.tau_on)
            pinned = math.exp(-self.alpha * T)
            factor += w[k] * pinned
            rate += w[k] * dT * pinned
        return factor + w[-1], rate  # the static exterior has T = 0

    def radii(self, xi: float, N: int) -> MapValues:
        """`X = B sum_k w_k e^(-alpha T_k)` and `d_xi X = -alpha B sum_k w_k dT_k/dxi e^(-alpha T_k)`."""
        B, _ = self.base.radii(0.0, N)  # the base is static, so its time argument is immaterial
        factor, rate = self._factors(xi, fractions(N))
        return B * factor, -self.alpha * B * rate

    def radius_at(self, xi: float, u: FloatArray) -> FloatArray:
        """`X(xi, u) = B(u) sum_k w_k(u) e^(-alpha T_k)`."""
        return self.base.radius_at(0.0, u) * self._factors(xi, u)[0]

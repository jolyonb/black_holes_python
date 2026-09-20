"""Time stepping (paper Section 7.6): RK4 in deviation form, the Courant step, and the step cap.

The method of lines: `calc_derivs` of `equations.py` turns the state into its rate, and this module advances it in
time with the classical four-stage Runge-Kutta method (RK4) at a fixed Courant number, the three-stage
strong-stability-preserving method (SSPRK3) being kept as a switch. Section 7.6 says why not the alternatives: an
adaptive embedded pair hunts for a stability boundary that a fixed Courant number already respects and costs more
per Courant-limited step; SSPRK3 certifies nothing here, since forward Euler is unstable with the production
kernels; leapfrog's stability region is the imaginary axis alone and the operator has real parts.

Deviation form. The integrator does not hold the state `y` but its deviation from FRW, `delta y = y - y_FRW(xi)`,
and its right-hand side is

    d_xi delta y = rate(xi, y_FRW(xi) + delta y) - d_xi y_FRW(xi),

with `y_FRW` and its rate known in closed form from the geometry (`state.py`). On a static map the two forms
coincide; on a moving one this keeps the far zone FRW to round-off where advancing `y` itself keeps it only to the
Runge-Kutta truncation of the map's motion (Section 7.6: `1e-15` against `1e-8` over one e-fold). It is always on.

The step. `Delta xi = min(C_CFL min_c Delta X_c / Lambda_hat_c, Delta xi_max)` with `C_CFL = 0.75` and
`Lambda_hat_c = max(Lambda_j, Lambda_j+1)` (eq:num:cfl): the shortest time in which the faster of a cell's two signal
speeds crosses the cell, recomputed every step, and a fixed cap (eq:num:stepcap) that binds only during the first
stretch of an early start, where the Courant step is of order one in `xi` and the growing mode, not sound, sets the
accuracy. The cap is derived from the local error of RK4 on
`y' = lambda_g y` accumulated over a super-horizon stretch, `kappa = (120 tol / (lambda_g T_sh))^(1/4)`, and is
`0.13` for `tol = 1e-5`, `T_sh = 4`. Steps are clipped to land on output times by the driver, not here.

`Frame` bundles what a stage needs at one time, the geometry, the background and the stencil weights, and `Scheme`
builds frames from the map and the layout: once for a static map, at every stage time for a moving one (Section 7.1).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction

import numpy as np

from pbh.eos import Background, EquationOfState
from pbh.equations import DerivsResult, calc_derivs
from pbh.geometry import Geometry
from pbh.kernels import KernelSettings
from pbh.layout import Layout
from pbh.maps import Map
from pbh.outer import OuterClosure
from pbh.state import frw_rate, frw_state
from pbh.stencils import FaceClosure, StencilWeights
from pbh.types import FloatArray

#: The Courant number of eq:num:cfl. The stable limit is about 0.95 for RK4 on the production footprint.
COURANT_NUMBER = 0.75


@dataclass(frozen=True)
class ButcherTableau:
    """An explicit Runge-Kutta method as its Butcher tableau, `c | a` over `b`, with exact rational entries.

    One step is `y + dxi sum_i b_i k_i` with `k_i = f(xi + c_i dxi, y + dxi sum_{j<i} a_ij k_j)`: the stages are
    evaluated in order, each from the ones before it, which is what "explicit" means and why row `i` of `a` has
    `i` entries.

    Attributes:
        c: The stage times as fractions of the step, one per stage; `c_0 = 0`.
        a: The stage weights, row `i` holding the `i` coefficients of the earlier stages.
        b: The final weights, one per stage, summing to one.
    """

    c: tuple[Fraction, ...]
    a: tuple[tuple[Fraction, ...], ...]
    b: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        """A tableau must be square, explicit and consistent: `c_i = sum_j a_ij` and `sum_i b_i = 1`."""
        stages = len(self.c)
        if len(self.a) != stages or len(self.b) != stages:
            raise ValueError("the tableau must have one row of a, one c and one b per stage")
        for i, row in enumerate(self.a):
            if len(row) != i:
                raise ValueError(f"row {i} of a must have {i} entries for an explicit method, got {len(row)}")
            if sum(row, Fraction(0)) != self.c[i]:
                raise ValueError(f"stage {i} is inconsistent: c_{i} = {self.c[i]} but its row of a sums to {sum(row)}")
        if sum(self.b, Fraction(0)) != 1:
            raise ValueError(f"the final weights must sum to one, got {sum(self.b)}")

    @property
    def stages(self) -> int:
        """The number of stages, hence of right-hand-side evaluations per step."""
        return len(self.c)


_HALF, _THIRD, _QUARTER, _SIXTH = Fraction(1, 2), Fraction(1, 3), Fraction(1, 4), Fraction(1, 6)

#: The classical fourth-order method: the production integrator (Section 7.6).
RK4 = ButcherTableau(
    c=(Fraction(0), _HALF, _HALF, Fraction(1)),
    a=(
        (),
        (_HALF,),
        (Fraction(0), _HALF),
        (Fraction(0), Fraction(0), Fraction(1)),
    ),
    b=(_SIXTH, _THIRD, _THIRD, _SIXTH),
)

#: The three-stage strong-stability-preserving method of Shu and Osher, in Butcher form: a switch (Section 7.6).
SSPRK3 = ButcherTableau(
    c=(Fraction(0), Fraction(1), _HALF),
    a=(
        (),
        (Fraction(1),),
        (_QUARTER, _QUARTER),
    ),
    b=(_SIXTH, _SIXTH, 2 * _THIRD),
)


class Integrator(Enum):
    """The time integrator (Section 7.6, Table tab:num:params)."""

    RK4 = "rk4"
    """The classical four-stage method at Courant number 0.75: the production choice."""

    SSPRK3 = "ssprk3"
    """The three-stage strong-stability-preserving method, admissible at Courant numbers 0.3 to 0.5; a switch."""

    @property
    def tableau(self) -> ButcherTableau:
        """The Butcher tableau of this integrator."""
        return RK4 if self is Integrator.RK4 else SSPRK3


@dataclass(frozen=True)
class Frame:
    """What a stage needs at one time: the geometry, the background and the stencil weights."""

    geo: Geometry
    bg: Background
    w: StencilWeights


@dataclass(frozen=True)
class Scheme:
    """Everything fixed over a stretch of the run, and the frame at any time in it.

    Attributes:
        eos: The equation of state.
        map: The map that places the faces in the scaled areal radius.
        layout: Which entries are unknowns; its `N` is the number of cells the map is evaluated for.
        closure: The excision-face closure the stencil weights are built for.
        outer: The outer closure.
        settings: The kernel switches.
    """

    eos: EquationOfState
    map: Map
    layout: Layout
    closure: FaceClosure
    outer: OuterClosure
    settings: KernelSettings
    _static_frame: tuple[Geometry, StencilWeights] | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """On a static map the geometry and the weights are computed once here and reused (Section 7.1)."""
        cached = (self._geometry_and_weights(0.0)) if self.map.is_static else None
        object.__setattr__(self, "_static_frame", cached)

    def _geometry_and_weights(self, xi: float) -> tuple[Geometry, StencilWeights]:
        geo = Geometry.of(*self.map.radii(xi, self.layout.N))
        return geo, StencilWeights.of(geo, self.layout, self.closure)

    def frame(self, xi: float) -> Frame:
        """The frame at time `xi`: the cached geometry on a static map, a fresh one on a moving map."""
        geo, w = self._static_frame if self._static_frame is not None else self._geometry_and_weights(xi)
        return Frame(geo=geo, bg=Background.at(self.eos, xi), w=w)

    def evaluate(self, xi: float, y: FloatArray) -> DerivsResult:
        """The time derivatives at time `xi` for the packed state `y`, with the fields they came from."""
        f = self.frame(xi)
        return calc_derivs(self.layout.unpack(y), f.geo, f.bg, self.eos, f.w, self.outer, self.settings)

    def frw(self, xi: float) -> FloatArray:
        """The packed FRW state at time `xi`."""
        return self.layout.pack(frw_state(self.frame(xi).geo, self.layout.j_e))

    def frw_rate(self, xi: float) -> FloatArray:
        """The packed rate of the FRW state at time `xi`."""
        return self.layout.pack(frw_rate(self.frame(xi).geo, self.layout.j_e))

    def deviation_rate(self, xi: float, dy: FloatArray) -> FloatArray:
        """The right-hand side in deviation form: the stage's rate on `y_FRW + delta y`, minus the FRW rate."""
        return self.layout.pack(self.evaluate(xi, self.frw(xi) + dy).rate) - self.frw_rate(xi)


type Rate = Callable[[float, FloatArray], FloatArray]
"""A right-hand side `f(xi, y)`."""


def explicit_rk_step(tableau: ButcherTableau, f: Rate, xi: float, y: FloatArray, dxi: float) -> FloatArray:
    """One step of the explicit Runge-Kutta method with this tableau, `y(xi) -> y(xi + dxi)`."""
    k: list[FloatArray] = []
    for c_i, a_i in zip(tableau.c, tableau.a, strict=True):
        y_i = y.copy()
        for a_ij, k_j in zip(a_i, k, strict=True):
            if a_ij:
                y_i += dxi * float(a_ij) * k_j
        k.append(f(xi + float(c_i) * dxi, y_i))
    return y + dxi * sum(float(b_i) * k_i for b_i, k_i in zip(tableau.b, k, strict=True))


def advance(scheme: Scheme, integrator: Integrator, xi: float, dy: FloatArray, dxi: float) -> FloatArray:
    """Advance the deviation `delta y` from `xi` to `xi + dxi` with the chosen integrator, in deviation form."""
    return explicit_rk_step(integrator.tableau, scheme.deviation_rate, xi, dy, dxi)


def courant_step(result: DerivsResult, geo: Geometry, layout: Layout, courant_number: float) -> float:
    """The Courant step `C_CFL min_c Delta X_c / Lambda_hat_c` over the retained cells (eq:num:cfl, first term).

    `Lambda_hat_c = max(Lambda_j, Lambda_j+1)` is the faster signal speed at the cell's two faces, so the ratio is the
    time that signal takes to cross the cell. On FRW at the start on the uniform grid this gives the first step
    `C_CFL Delta X / c_s(xi_0)`, of order one in `xi` on a grid that resolves a super-horizon perturbation.
    """
    cells = layout.cells
    Lam = result.speeds.Lam
    fastest = np.maximum(Lam[cells.start : cells.stop], Lam[cells.start + 1 : cells.stop + 1])
    crossing = float(np.min(geo.dX[cells] / fastest))
    if not math.isfinite(crossing) or crossing <= 0.0:
        raise ValueError(f"the shortest cell crossing time is {crossing!r}: no Courant step can be formed")
    return courant_number * crossing


class StepLimit(Enum):
    """Which of the two limits of eq:num:cfl set the step."""

    COURANT = "courant"
    """The Courant step: the shortest cell crossing time."""

    CAP = "cap"
    """The fixed cap `Delta xi_max` of eq:num:stepcap."""


@dataclass(frozen=True)
class StepChoice:
    """The step eq:num:cfl chose, and which limit chose it (the output spec records the limit every step)."""

    dxi: float
    limit: StepLimit


def step_size(result: DerivsResult, geo: Geometry, layout: Layout, courant_number: float, cap: float) -> StepChoice:
    """The step of eq:num:cfl: the smaller of the Courant step and the cap, with the limit that bound.

    Clipping to an output time is the driver's business and is recorded there as a third kind of limit.
    """
    courant = courant_step(result, geo, layout, courant_number)
    if courant <= cap:
        return StepChoice(dxi=courant, limit=StepLimit.COURANT)
    return StepChoice(dxi=cap, limit=StepLimit.CAP)


def step_cap(eos: EquationOfState, tolerance: float = 1e-5, super_horizon_efolds: float = 4.0) -> float:
    """The fixed cap `Delta xi_max = kappa / lambda_g` on the step (eq:num:stepcap).

    RK4's local error on `y' = lambda_g y` is `(lambda_g Delta xi)^5 / 120` per step, so the relative error of the
    growing amplitude accumulated over a stretch of `T_sh` e-folds is `T_sh lambda_g (lambda_g Delta xi)^4 / 120`;
    requiring it below the tolerance gives `kappa = (120 tol / (lambda_g T_sh))^(1/4)`, `0.13` for radiation at
    `tol = 1e-5`, `T_sh = 4`, and `0.12` at `T_sh = 6`.
    """
    lambda_g = eos.growing_mode_rate
    kappa = (120.0 * tolerance / (lambda_g * super_horizon_efolds)) ** 0.25
    return kappa / lambda_g

"""Time stepping (paper Section 7.6): RK4 in deviation form, the Courant step, and the step cap.

The method of lines: `calc_derivs` of `equations.py` turns the state into its rate, and this module advances it in
time with the classical four-stage Runge-Kutta method (RK4) at a fixed Courant number. Section 7.6 says why not the
alternatives: an adaptive embedded pair hunts for a stability boundary that a fixed Courant number already respects
and costs more per Courant-limited step; the three-stage strong-stability-preserving method (SSPRK3) certifies no
stability here, since forward Euler is unstable with the production kernels, and its positivity condition bought
nothing measurable on the near-threshold ladder; leapfrog's stability region is the imaginary axis alone and the
operator has real parts.

Deviation form. The integrator does not hold the state `y` but its deviation from FRW, `delta y = y - y_FRW(xi)`,
and its right-hand side is

    d_xi delta y = rate(xi, y_FRW(xi) + delta y) - d_xi y_FRW(xi),

with `y_FRW` and its rate known in closed form from the geometry (`state.py`). On a static map the two forms
coincide; on a moving one this keeps the far zone FRW to round-off where advancing `y` itself keeps it only to the
Runge-Kutta truncation of the map's motion (Section 7.6: `1e-15` against `1e-8` over one e-fold). It is always on. A
cell far below the background stores its content whole instead (`storage.py`), and the integrator advances each stored
number by its own rate, `DerivsResult.stored_rate`; the flags change only at step boundaries, so a step's stages all
share them.

The step. `Delta xi = min(C_CFL min_c Delta X_c / Lambda_hat_c, Delta xi_max)` with `C_CFL = 0.75` and
`Lambda_hat_c = max(Lambda_j, Lambda_j+1)` (eq:num:cfl): the shortest time in which the faster of a cell's two signal
speeds crosses the cell, recomputed every step, and a fixed cap (eq:num:stepcap) that binds only during the first
stretch of an early start, where the Courant step is of order one in `xi` and the growing mode, not sound, sets the
accuracy. The cap is derived from the local error of RK4 on
`y' = lambda_g y` accumulated over a super-horizon stretch, `kappa = (120 tol / (lambda_g T_sh))^(1/4)`, and is
`0.13` for `tol = 1e-5`, `T_sh = 4`. Steps are clipped to land on output times by the driver, not here.

The checked step. Every stage input and the result of a step must be finite and inside the hyperbolic domain, every
retained `rho_c > 0` and `Gammabar_j^2 > 0`; an attempt that fails is refused and the step retried from the same state
with half the step (`advance_checked`). This is what keeps accepted states positive: the semi-discrete scheme is
positive (eq:num:positivity), but an explicit step can overshoot it.

The clock. The time is advanced by adding the step to it, and a step of a few ulps of `xi` is rounded away in the sum:
deep in a void the step falls to `1e-14` at `xi ~ 6`, and the clock would stop while the state moves on. A step shorter
than `COMPENSATED_BELOW |xi|` is therefore added with its rounding error carried to the next (`Clock`, the two-sum of
Knuth), so the clock keeps the sum of the steps however small they become. Ordinary runs never take a step that short,
and their clock is the plain sum, to the bit.

`Frame` bundles what a stage needs at one time, the geometry, the stencil weights, the scale factor and the reference
solution, and `Scheme` builds frames from the map and the layout: once for a static map (bar the scale factor), at
every stage time for a moving one (Section 7.1), keeping the frames of the last few times, which the stepper and the
driver ask for repeatedly.
"""

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from pbh.derived import NotHyperbolicError
from pbh.eos import Background, EquationOfState, Spacetime
from pbh.equations import DerivsResult, calc_derivs
from pbh.geometry import Geometry
from pbh.kernels import KernelSettings
from pbh.layout import Layout
from pbh.maps import BlendMap, Map, MapValues
from pbh.outer import OuterClosure
from pbh.state import FrwReference, State
from pbh.stencils import StencilWeights
from pbh.storage import any_whole, deviation_of, whole_of
from pbh.types import BoolArray, FloatArray, read_only

if TYPE_CHECKING:  # the Rust engine is imported only when a Scheme asks for it (`Scheme.__post_init__`)
    from pbh_engine import StageFrame

    from pbh.rust_engine import RustStage

#: The Courant number of eq:num:cfl. RK4's stable limit on the production footprint is 0.865 (0.835 to x_max = 48).
COURANT_NUMBER = 0.75

#: A step shorter than this fraction of `|xi|`, some six thousand ulps, is added to the clock with its rounding carried
#: (`Clock`). An ordinary run never takes one: twenty halvings of an acoustic step of `1e-4` are `1e-10`, twenty times
#: longer at `xi = 6`. The deep voids take steps of `1e-14`.
COMPENSATED_BELOW = 2.0**-40


@dataclass(frozen=True)
class Clock:
    """The time `xi` and the rounding error `carry` of its last compensated additions (`xi + carry` is the sum).

    The carry is zero while every step is longer than `COMPENSATED_BELOW |xi|`, and the clock then the plain sum.
    """

    xi: float
    carry: float = 0.0

    def advanced(self, dxi: float) -> Clock:
        """The clock a step `dxi` later: the plain sum, or, for a short step or with a carry, the compensated one.

        The compensated sum adds `dxi + carry` and keeps the exact rounding error of that addition as the new carry
        (Knuth's two-sum); only the rounding of `dxi + carry` itself, a part in `1e16` of the step, is lost.
        """
        if self.carry == 0.0 and dxi >= COMPENSATED_BELOW * abs(self.xi):
            return Clock(self.xi + dxi)
        increment = dxi + self.carry
        xi = self.xi + increment
        back = xi - self.xi
        return Clock(xi, (self.xi - (xi - back)) + (increment - back))


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

    @functools.cached_property
    def floats(self) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...], tuple[float, ...]]:
        """`(c, a, b)` as floats, converted once: the stepper takes them at every step."""
        return tuple(map(float, self.c)), tuple(tuple(map(float, row)) for row in self.a), tuple(map(float, self.b))


_HALF, _THIRD, _SIXTH = Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)

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


class Engine(Enum):
    """Which implementation evaluates a stage: the numpy engine (the reference and the default) or the Rust engine.

    Both compute `calc_derivs` with the same operations in the same order (`rust/src`, one function per function of
    `equations.py` and what it calls), and a stage agrees between them to the bit on this machine on every non-NaN
    entry, with NaN in the same entries (tests/test_rust_engine.py). The Rust engine emits none of numpy's
    `RuntimeWarning`s, and the sign of a computed NaN may differ (`pbh.rust_engine` says why). The engine is not
    physics: it is chosen where a run is started (`pbh run --engine`), not configured, and a run records it in its
    files' provenance; a restart may switch.
    """

    PYTHON = "python"
    """numpy, `equations.calc_derivs`."""

    RUST = "rust"
    """The compiled stage of `pbh_engine`, through `pbh.rust_engine`: an optional package (`rust/`, the `rust`
    dependency group), which a Scheme on this engine imports and, when it is not installed, refuses by saying so."""


#: How many frames a `Scheme` keeps: the three stage times of an RK4 step, `xi`, `xi + dxi / 2` and `xi + dxi`, and
#: one more, the output time `land` that the driver clipped the step to, at which `checked_step` evaluates the result
#: while its fourth stage stays at `xi + dxi`.
FRAMES_KEPT = 4


@dataclass(frozen=True)
class Frame:
    """What a stage needs at one time: the geometry, the stencil weights, the scale factor and the reference solution.

    A frame is shared by every caller that asks its `Scheme` for the same time, so its arrays are read-only: an
    in-place write by any consumer raises rather than silently changing the next stage's geometry.

    Attributes:
        geo: The geometry.
        bg: The scale-factor background at this time: `a`, `H`, the Hubble radius, `Gammabar`, `c_s` and `h`.
        w: The stencil weights.
        reference: The reference solution `y_FRW` on this geometry (the fluid at rest in flat spacetime), unpacked,
            with its rate and flux.
        y_frw: `reference.state` packed by the layout: the vector to which the integrator adds its deviation.
        rust: The same frame as the Rust engine holds it, when the Scheme runs it (the Rust engine builds the frame,
            and the fields above hold its numbers); `None` on the numpy engine.
    """

    geo: Geometry
    bg: Background
    w: StencilWeights
    reference: FrwReference
    y_frw: FloatArray
    rust: StageFrame | None = None


@dataclass(frozen=True)
class Scheme:
    """Everything fixed over a stretch of the run, and the frame at any time in it.

    Attributes:
        eos: The equation of state.
        map: The map that places the faces in the scaled areal radius.
        layout: Which entries are unknowns; its `N` is the number of cells the map is evaluated for.
        outer: The outer closure.
        settings: The kernel switches.
        spacetime: The spacetime: FRW (production), or flat, the limit without gravity of Section 7.7, whose reference
            is the fluid at rest. Excision needs gravity, so a flat scheme has no excised cells.
        engine: Which implementation evaluates a stage: numpy (the default) or Rust (`Engine`).
    """

    eos: EquationOfState
    map: Map
    layout: Layout
    outer: OuterClosure
    settings: KernelSettings
    spacetime: Spacetime = Spacetime.FRW
    engine: Engine = Engine.PYTHON
    _rust: RustStage | None = field(init=False, repr=False, compare=False)
    _static_frame: Frame | None = field(init=False, repr=False, compare=False)
    _frames: dict[float, Frame] = field(init=False, default_factory=dict[float, Frame], repr=False, compare=False)
    _stage_frames: dict[float, tuple[StageFrame, Background]] = field(
        init=False, default_factory=dict[float, tuple["StageFrame", Background]], repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """On a static map the frame is built once here, and every later frame shares all of it but `bg`.

        The geometry and the weights do not depend on `xi` on a static map (Section 7.1), nor does the reference
        solution, whose only other input `h` depends on the spacetime alone.
        """
        if self.spacetime is Spacetime.FLAT and self.layout.j_e > 0:
            raise ValueError("flat spacetime has no gravity, so no black hole to excise")
        rust = None
        if self.engine is Engine.RUST:
            try:
                from pbh import rust_engine  # the extension is loaded only by a Scheme that runs it
            except ModuleNotFoundError as error:
                if error.name != "pbh_engine":
                    raise
                raise ModuleNotFoundError(
                    "the Rust engine was asked for, but pbh_engine is not installed: install it with "
                    "`uv sync --group rust` (a Rust toolchain is required), or run on the numpy engine, the default",
                    name=error.name,
                ) from error
            rust = rust_engine.RustStage(self.eos, self.settings, self.outer, self.layout)
        object.__setattr__(self, "_rust", rust)
        object.__setattr__(self, "_static_frame", self._new_frame(0.0) if self.map.is_static else None)

    def __reduce__(
        self,
    ) -> tuple[type[Scheme], tuple[EquationOfState, Map, Layout, OuterClosure, KernelSettings, Spacetime, Engine]]:
        """Pickle and copy a Scheme as its settings, from which the copy rebuilds its frames when it needs them.

        The frames are a cache, and on the Rust engine they hold the extension's objects, which cannot be pickled;
        rebuilding gives the same frames to the bit, since they are computed afresh from the same settings. So a Scheme
        on either engine can be sent to a worker process or deep-copied.
        """
        return (Scheme, (self.eos, self.map, self.layout, self.outer, self.settings, self.spacetime, self.engine))

    def _new_frame(self, xi: float) -> Frame:
        """The frame at time `xi` built from the map: every stage time on a moving map, once on a static one."""
        X, X_xi = self._radii(xi)
        bg = Background.at(self.eos, xi, self.spacetime)
        rust = None
        if self._rust is None:
            geo = Geometry.of(X, X_xi)
            w = StencilWeights.of(geo, self.layout)
            reference = FrwReference.of(geo, self.eos, self.layout.j_e, bg.hubble)  # makes its own arrays read-only
        else:  # built in Rust, the same numbers (`RustStage.frame`)
            geo, w, reference, rust = self._rust.frame(X, X_xi, bg.hubble)
        y_frw = self.layout.pack(reference.state)
        # Shared by every stage and caller that asks for this time (`frame`): read-only, so a stray write raises.
        read_only(geo.X, geo.X_xi, geo.dV, geo.dV_xi, geo.sbar, geo.dS, geo.dX, geo.Xm, geo.X2, geo.X3)
        read_only(geo.s_in, geo.s_out, w.grad_s, w.centred_U, w.r_L, w.r_R, y_frw)
        return Frame(geo=geo, bg=bg, w=w, reference=reference, y_frw=y_frw, rust=rust)

    def _radii(self, xi: float) -> MapValues:
        """The map at the faces at `xi`: on the Rust engine a blend map's are formed in Rust, the same numbers."""
        if self._rust is not None and isinstance(self.map, BlendMap):
            return self._rust.blend_radii(self.map, xi)
        return self.map.radii(xi, self.layout.N)

    def frame(self, xi: float) -> Frame:
        """The frame at time `xi`, kept for the last `FRAMES_KEPT` distinct times.

        On a static map every frame is the one built in `__post_init__` with the background of its own time; on a
        moving map each new time builds its own. Frames are keyed on the exact float `xi`: a time that differs in the
        last bit is simply a new frame. Everything a frame depends on besides `xi` is fixed for the Scheme's lifetime,
        and a re-excision or a new zone builds a new Scheme, so a kept frame is never stale.
        """
        frame = self._frames.get(xi)
        if frame is None:
            static = self._static_frame
            if static is None:
                frame = self._new_frame(xi)
            else:
                bg = Background.at(self.eos, xi, self.spacetime)
                frame = Frame(
                    geo=static.geo,
                    bg=bg,
                    w=static.w,
                    reference=static.reference,
                    y_frw=static.y_frw,
                    rust=static.rust,
                )
            if len(self._frames) >= FRAMES_KEPT:
                del self._frames[next(iter(self._frames))]  # dicts keep insertion order: this is the oldest
            self._frames[xi] = frame
        return frame

    def evaluate(self, xi: float, y: FloatArray) -> DerivsResult:
        """The time derivatives at time `xi` for the packed state `y`, with the fields they came from."""
        f = self.frame(xi)
        if self._rust is not None:
            assert f.rust is not None
            return self._rust.evaluate(f.rust, f.bg, y)
        state = self.layout.unpack(y)
        return calc_derivs(state, f.geo, f.bg, self.eos, f.w, self.outer, self.settings, reference=f.reference)

    def evaluate_deviation(self, xi: float, dy: FloatArray, whole: BoolArray | None = None) -> DerivsResult:
        """`evaluate` at the state `y_FRW + delta y`, handing the stage the deviation as well, which it needs whole.

        The sum `y_FRW + delta y` rounds each entry to the size of its FRW value; `Gammabar^2` is formed from the
        deviation instead, which has not been rounded (see `derive`). The cells `whole` marks hold their content
        itself (`storage.py`).
        """
        f = self.frame(xi)
        whole = any_whole(whole)
        if self._rust is not None:
            assert f.rust is not None
            return self._rust.evaluate_deviation(f.rust, f.bg, dy, whole)
        stored = self.layout.unpack(dy)
        state = self.whole_state(xi, stored, whole)
        return calc_derivs(
            state,
            f.geo,
            f.bg,
            self.eos,
            f.w,
            self.outer,
            self.settings,
            deviation=deviation_of(stored, whole, f.reference.state.E),
            reference=f.reference,
            whole=whole,
        )

    def rust_attempt(
        self,
        stages: list[Stage],
        times: list[float],
        arrive: float,
        dy: FloatArray,
        dxi: float,
        a: tuple[tuple[float, ...], ...],
        b: tuple[float, ...],
        whole: BoolArray | None = None,
    ) -> Attempt:
        """`checked_step`'s attempt on the Rust engine, from the first stage `stages[0]` (`pbh.rust_engine`).

        The stages at `times` and the result at `arrive` are formed and checked in Rust, with the tableau `a`, `b`, and
        the cells `whole` marks stored whole.
        """
        assert self._rust is not None
        arrival = self.frame(arrive)  # held whole: the driver reads the state arrived at through it
        assert arrival.rust is not None
        at = [(arrival.rust, arrival.bg) if t == arrive else self._stage_time(t) for t in times]
        return self._rust.attempt(stages, times, at, (arrival.rust, arrival.bg), dy, dxi, a, b, whole)

    def _stage_time(self, xi: float) -> tuple[StageFrame, Background]:
        """The Rust engine's frame and the background at `xi`, for a stage inside a step and nothing else.

        A frame held for this time, or the static map's, is used as it is. Otherwise only the Rust frame is built, not
        the Python's `Geometry`, `StencilWeights` and `FrwReference`, which nothing reads at a time inside a step; it is
        kept, as frames are, for the last `FRAMES_KEPT` such times (a refused step retries at other times).
        """
        assert self._rust is not None
        if xi in self._frames or self._static_frame is not None:
            frame = self.frame(xi)
            assert frame.rust is not None
            return frame.rust, frame.bg
        kept = self._stage_frames.get(xi)
        if kept is None:
            X, X_xi = self._radii(xi)
            bg = Background.at(self.eos, xi, self.spacetime)
            kept = (self._rust.stage_frame(X, X_xi, bg.hubble), bg)
            if len(self._stage_frames) >= FRAMES_KEPT:
                del self._stage_frames[next(iter(self._stage_frames))]
            self._stage_frames[xi] = kept
        return kept

    def whole_state(self, xi: float, deviation: State, whole: BoolArray | None = None) -> State:
        """The state `y_FRW + delta y` at time `xi`, from the unpacked deviation: the one recipe for it.

        Bit for bit `unpack(frw(xi) + delta y)`, without packing and unpacking a second vector: the retained entries
        are the same sums; the excised ones are NaN either way (the deviation's are); and where `unpack` sets a value
        by fiat, the sum gives it exactly, `U_0 = h X_0 + 0 = 0` (`Geometry.of` insists on `X_0 = 0`) and before
        excision `M_e = X_0^3 + 0 = 0`. A cell `whole` marks holds its content itself, which is taken as it is.
        """
        return whole_of(deviation, any_whole(whole), self.frame(xi).reference.state)

    def frw(self, xi: float) -> FloatArray:
        """The packed background state at time `xi`: FRW, or the fluid at rest in flat spacetime (read-only)."""
        return self.frame(xi).y_frw

    def deviation_rate(self, xi: float, dy: FloatArray) -> FloatArray:
        """The right-hand side in deviation form: the stage's rate of the deviation at `y_FRW + delta y`."""
        return self.layout.pack(self.evaluate_deviation(xi, dy).deviation_rate)


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


def advance(scheme: Scheme, xi: float, dy: FloatArray, dxi: float) -> FloatArray:
    """Advance the deviation `delta y` from `xi` to `xi + dxi` by one unchecked RK4 step, in deviation form."""
    return explicit_rk_step(RK4, scheme.deviation_rate, xi, dy, dxi)


@dataclass(frozen=True)
class StageFluxes:
    """The scalars of a stage that the step's bookkeeping needs; free to collect.

    The total mass obeys `d_xi M_total = (2 - 3 alpha) M_total - 3 F_N` exactly. The outer face is static, so the FRW
    parts of both sides cancel identically, `(2 - 3 alpha) X_N^3 = 3 alpha w X_N^3`, and the bookkeeping is checked in
    the deviations, `d_xi delta M_total = (2 - 3 alpha) delta M_total - 3 delta F_N`, where it is limited by the
    rounding of the deviation rather than of `M_total` itself.

    Attributes:
        F_N: The outer flux; `F_je` the flux through the excision face (`0` unexcised).
        M_total: The total mass in the domain, `M_N = M_e + 3 sum E_c`.
        delta_F_N: The outer flux less its FRW value.
        delta_M_total: The total mass less its FRW value `X_N^3`, the cumulative sum of the deviations.
    """

    F_N: float
    F_je: float
    M_total: float
    delta_F_N: float
    delta_M_total: float

    @classmethod
    def of(cls, result: DerivsResult, layout: Layout) -> StageFluxes:
        """Collect the stage's fluxes and total mass."""
        N, j_e = layout.N, layout.j_e
        d = result.derived
        return cls(
            F_N=float(result.F[N]),
            F_je=float(result.F[j_e]) if j_e > 0 else 0.0,
            M_total=float(d.M[N]),
            delta_F_N=float(result.delta_F[N]),
            delta_M_total=float(d.delta_M[N]),
        )


@dataclass(frozen=True)
class Stage:
    """One stage of a step as the step's record reads it: its time, its fluxes, and its packed deviation rate `k`.

    The stage's whole result is not kept: the record needs only these (`StageFluxes`, and the last stage's rate for
    RK4's companion estimate), and on the Rust engine the stages after the first never become Python objects.
    """

    xi: float
    fluxes: StageFluxes
    k: FloatArray


# --- the checked step: every stage and the result inside the hyperbolic domain, or a smaller step (Section 7.6) ---

#: The most halvings of one step before the run aborts: a factor of about 1e6, from an acoustic step of about 1e-2 down
#: to about 1e-8, far above the round-off of `xi`.
MAX_HALVINGS = 20
#: The most halvings a failure of `Gammabar^2 > 0` gets. A stage or result with `Gammabar^2 <= 0` is either an overshoot
#: of a thin margin, which a shorter step cures at once, or the flow itself reaching `Gamma = 0`, a trapped region the
#: excision has not caught, which no step cures: two halvings tell the two apart without creeping.
CHART_HALVINGS = 2


class FailureCause(Enum):
    """Why an attempted step was refused: where (a stage or the result) and what (Section 7.6)."""

    STAGE_NONFINITE = "stage_nonfinite"
    STAGE_RHO = "stage_rho"
    STAGE_GAMMABAR2 = "stage_Gammabar2"
    RESULT_NONFINITE = "result_nonfinite"
    RESULT_RHO = "result_rho"
    RESULT_GAMMABAR2 = "result_Gammabar2"

    @property
    def is_chart(self) -> bool:
        """Whether the failure is the areal chart's, `Gammabar^2 <= 0`, not positivity's or a non-finite value."""
        return self in (FailureCause.STAGE_GAMMABAR2, FailureCause.RESULT_GAMMABAR2)


@dataclass(frozen=True)
class StepFailure:
    """A refused attempt: its cause, stage, and the cell or face that failed.

    Attributes:
        cause: Why it was refused.
        stage: `2`, `3`, ... for a stage input, `0` for the result.
        index: The cell (density; `N` for the outer face's density) or face (`Gammabar^2`) that failed; `-1` for a
            non-finite state found before any evaluation.
        value: The failing value, NaN if none.
    """

    cause: FailureCause
    stage: int
    index: int
    value: float


@dataclass(frozen=True)
class Attempt:
    """One attempted step: on success the deviation arrived at, every stage, and the rate there; else the failure."""

    dy: FloatArray | None
    stages: list[Stage]
    result: DerivsResult | None
    failure: StepFailure | None


def checked_step(
    scheme: Scheme,
    xi: float,
    dy: FloatArray,
    dxi: float,
    first: DerivsResult,
    land: float | None = None,
    whole: BoolArray | None = None,
) -> Attempt:
    """One RK4 step in deviation form, every stage and the result checked.

    Every stage input after the first and the result must be finite, and their evaluation must find them inside the
    hyperbolic domain, every retained `rho_c > 0` and `Gammabar_j^2 > 0` (`derive` asserts both). The first stage is
    the accepted state, whose evaluation `first` the caller already holds. The result is evaluated here, so that a
    result outside the domain is caught like a stage, and its rate is handed back as the next step's first stage. A
    `Gammabar^2` that evaluates to NaN or an infinity is a non-finite value, not a failure of the chart. The checks,
    not the method, keep accepted states positive.
    `land`, if given, is the time the step arrives at, an output time the driver clipped it to or where the compensated
    clock arrives (`Clock`): the result is evaluated there exactly, as a restart from that time evaluates it, rather
    than at the rounded `xi + dxi`.
    `whole` marks the cells stored whole (`storage.py`), whose stored content advances by its whole rate.
    """
    c, a, b = RK4.floats
    layout = scheme.layout
    stages = [Stage(xi=xi, fluxes=StageFluxes.of(first, layout), k=layout.pack(first.stored_rate))]
    arrive = xi + dxi if land is None else land
    whole = any_whole(whole)
    if scheme.engine is Engine.RUST:
        return scheme.rust_attempt(stages, [xi + c_i * dxi for c_i in c[1:]], arrive, dy, dxi, a, b, whole)
    rate = scheme.evaluate_deviation if whole is None else functools.partial(scheme.evaluate_deviation, whole=whole)

    def evaluate(xi_i: float, dy_i: FloatArray, stage: int) -> tuple[DerivsResult | None, StepFailure | None]:
        where = "stage" if stage else "result"
        if not bool(np.all(np.isfinite(dy_i))):
            return None, StepFailure(FailureCause(f"{where}_nonfinite"), stage, -1, math.nan)
        try:
            return rate(xi_i, dy_i), None
        except NotHyperbolicError as e:
            what = e.field if math.isfinite(e.value) else "nonfinite"
            return None, StepFailure(FailureCause(f"{where}_{what}"), stage, e.index, e.value)

    for n, (c_i, a_i) in enumerate(zip(c, a, strict=True)):
        if n == 0:
            continue  # the accepted state's, held by the caller
        dy_i = dy.copy()
        for a_ij, stage in zip(a_i, stages, strict=True):
            if a_ij:
                dy_i += dxi * a_ij * stage.k
        xi_i = xi + c_i * dxi
        result, failure = evaluate(xi_i, dy_i, n + 1)
        if result is None:
            return Attempt(None, stages, None, failure)
        stages.append(Stage(xi=xi_i, fluxes=StageFluxes.of(result, layout), k=layout.pack(result.stored_rate)))
    dy_new = dy + dxi * sum(b_i * stage.k for b_i, stage in zip(b, stages, strict=True))
    result, failure = evaluate(arrive, dy_new, 0)
    return Attempt(dy_new if result is not None else None, stages, result, failure)


@dataclass(frozen=True)
class AcceptedStep:
    """A step the checks accepted.

    Attributes:
        dxi: The step taken, halved as often as attempts were refused.
        dy: The deviation arrived at.
        result: The rate there, the next step's first stage.
        stages: The accepted attempt's stages.
        refused: The attempts refused before it, in order.
        clock: The clock arrived at, the time at which `result` was evaluated.
    """

    dxi: float
    dy: FloatArray
    result: DerivsResult
    stages: list[Stage]
    refused: list[StepFailure]
    clock: Clock


class StepAbortError(Exception):
    """No step the retry policy allows passed the checks: the run ends, and the last failure says why."""

    def __init__(self, failure: StepFailure, refused: list[StepFailure]) -> None:
        super().__init__(f"{failure.cause.value} at index {failure.index}")
        self.failure = failure
        self.refused = refused


def advance_checked(
    scheme: Scheme,
    xi: float,
    dy: FloatArray,
    dxi: float,
    first: DerivsResult,
    land: float | None = None,
    carry: float = 0.0,
    whole: BoolArray | None = None,
) -> AcceptedStep:
    """The checked step with its retry policy (Section 7.6).

    A refused attempt is retried from the same state with half the step, at most `MAX_HALVINGS` times, and at most
    `CHART_HALVINGS` times for `Gammabar^2`; beyond either the run ends (`StepAbortError`). The positivity of accepted
    states rests on the checks; that some step passes rests on the semi-discrete scheme's
    (eq:num:positivity) and on the stages tending to the accepted state as the step shrinks. `land` is the full step's
    arrival time (`checked_step`); a halved step lands short, where the clock `Clock(xi, carry)` arrives. The accepted
    step carries the clock it arrived at, the one the result was evaluated at. `whole` marks the cells stored whole.
    """
    refused: list[StepFailure] = []
    chart = 0
    clock = Clock(xi, carry)
    while True:
        arrived = Clock(land) if land is not None and not refused else clock.advanced(dxi)
        attempt = checked_step(scheme, xi, dy, dxi, first, arrived.xi, whole)
        if attempt.failure is None:
            assert attempt.dy is not None
            assert attempt.result is not None
            return AcceptedStep(dxi, attempt.dy, attempt.result, attempt.stages, refused, arrived)
        refused.append(attempt.failure)
        chart += attempt.failure.cause.is_chart
        if len(refused) > MAX_HALVINGS or chart > CHART_HALVINGS:
            raise StepAbortError(attempt.failure, refused)
        dxi *= 0.5


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


def step_cap(eos: EquationOfState, tolerance: float = 1e-7, super_horizon_efolds: float = 4.0) -> float:
    """The fixed cap `Delta xi_max = kappa / lambda_g` on the step (eq:num:stepcap).

    RK4's local error on `y' = lambda_g y` is `(lambda_g Delta xi)^5 / 120` per step, so the relative error of the
    growing amplitude accumulated over a stretch of `T_sh` e-folds is `T_sh lambda_g (lambda_g Delta xi)^4 / 120`;
    requiring it below the tolerance gives `kappa = (120 tol / (lambda_g T_sh))^(1/4)`, `0.13` for radiation at
    `tol = 1e-5`, `T_sh = 4`, and `0.12` at `T_sh = 6`; the default `1e-7` gives `0.042`, since at `1e-5` the cap's
    amplitude error moved the threshold by a few parts in 1e6 (analysis experiments e9, e10) for at most a few per
    cent more steps. Flat spacetime has no growing mode, so a flat run passes no
    cap (`math.inf`) to `step_size`.
    """
    lambda_g = eos.growing_mode_rate
    kappa = (120.0 * tolerance / (lambda_g * super_horizon_efolds)) ** 0.25
    return kappa / lambda_g

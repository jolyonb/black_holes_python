"""The Rust engine against the numpy engine: the same stage, field by field, and the switch that chooses it.

The numpy engine is the reference. The Rust engine (`rust/src`, through `pbh.rust_engine`) computes every field of
`DerivsResult` with the same operations in the same order, so the comparison asks for the values themselves, not a
tolerance:

* at `w = 1/3` and `w = 1` the stage uses only `+ - * /`, `sqrt` and `abs`, which IEEE 754 rounds correctly on every
  platform, and the extrema, which are exact: every entry is equal by `==` and the NaN entries are the same entries.
  (The sign of a zero and a NaN's sign and payload are not asserted. numpy's SIMD extrema order `+0` and `-0`
  differently on different processors, and `rust/src/numpy_like.rs` fixes the arm64 convention. A NaN computed inside
  the stage may carry the opposite sign bit, because the compiler folds a negation of it into a neighbouring
  subtraction, which is exact for every number. On the machine the engine was written on, every non-NaN entry agrees
  byte for byte, signed zeros included, which the scratch harness of the Rust work asserts.)
* at any other `w` the lapse goes through `pow`, `log1p` and `expm1`, which numpy may take from its own SIMD kernels
  on some processors, up to `LAPSE_ULPS` ulp from the C library's. Everything else the two engines compute by the
  same correctly rounded operations, so what can differ is exactly what that lapse error propagates to. The allowance
  is measured, entry by entry, by propagating it: the numpy stage is evaluated again with every lapse the stage forms
  (`e^phi` and `e^phi - 1`, at the cells, the outer face and the kernels' one-sided fluxes) moved by `LAPSE_ULPS` ulp,
  under four sign patterns, and an entry may differ by `LAPSE_MARGIN` times the largest response it shows, plus
  `LAPSE_ULPS` ulp of itself. So an entry that does not respond is held to a few ulp of itself, a small entry is
  measured against its own sensitivity rather than against the field's largest entry, and the fields that are formed
  without the lapse at all (`LAPSE_FREE`) are asserted exactly by name, after checking that none of them responds.
  Here too the measured difference is zero.

Refusals must agree as well: the same `NotHyperbolicError` (field, index, and the value, bit for bit or both NaN) and
the same `ValueError` of a closure, raised by the same call. The states are the violent ones of the positivity tests
(`violent.py`), on static and moving maps, before and after excision, with every kernel switch and closure, in FRW and
in flat spacetime.
"""

import ast
import copy
import importlib
import math
import pickle
import subprocess
import sys
from collections.abc import Callable
from dataclasses import astuple, dataclass, fields, is_dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

pytest.importorskip("pbh_engine", reason="the optional Rust engine is not installed (uv sync --group rust)")

import numpy as np
import pbh_engine
import yaml
from violent import FAMILIES, SEEDS, W_CASES, Case, violent_case

from pbh import rust_engine
from pbh.cli import main
from pbh.config import ConfigError, load, save
from pbh.derived import NotHyperbolicError
from pbh.driver import RunPaths
from pbh.eos import RADIATION, EquationOfState, Spacetime
from pbh.equations import DerivsResult
from pbh.horizon import Trapping, find_horizons, near_zone, near_zone_numbers, trapping
from pbh.kernels import CENTRED_SCHEME, PRODUCTION_KERNELS, DensityLimiter, KernelSettings, ViscousFlux
from pbh.layout import Layout
from pbh.maps import BlendMap, IdentityMap, Map, PinnedMap, SinhStretch, Zone
from pbh.michel import michel_flow, michel_grid, michel_state
from pbh.monitors import emptying_rates
from pbh.outer import (
    HeldAtFrw,
    HeldExterior,
    OuterClosure,
    OuterInputs,
    OuterRows,
    OutgoingWave,
    PenaltyStrengths,
)
from pbh.output import RunReader
from pbh.rust_engine import RustStage
from pbh.state import State
from pbh.timestep import (
    FRAMES_KEPT,
    RK4,
    AcceptedStep,
    Engine,
    Frame,
    Scheme,
    advance_checked,
    checked_step,
    step_cap,
    step_size,
)
from pbh.types import FloatArray

RAD = EquationOfState(RADIATION)
EPS = float(np.finfo(float).eps)
#: The general-w lapse error allowed, in ulp of `e^phi` and of `e^phi - 1`: numpy's SIMD `pow`, `log1p` and `expm1`
#: against the C library's (see the module docstring).
LAPSE_ULPS = 4.0
#: How far an entry may exceed the largest response to one sign pattern of the lapse error: each entry is formed from
#: the lapses of at most about four cells and faces, whose independent errors the four patterns do not cover jointly.
LAPSE_MARGIN = 4.0
#: The fields formed without the lapse, by `derive` (the density, the mass, `Gammabar^2`, the face densities) and by the
#: density reconstruction: equal at every `w`, since only correctly rounded operations reach them.
LAPSE_FREE = frozenset(
    {
        "derived.rho",
        "derived.delta_rho",
        "derived.M",
        "derived.delta_M",
        "derived.mt",
        "derived.delta_U",
        "derived.delta_m",
        "derived.Gammabar2",
        "derived.rho_f",
        "derived.delta_rho_f",
        "kernels.rho_L",
        "kernels.rho_R",
        "kernels.delta_rho_L",
        "kernels.delta_rho_R",
        "kernels.theta_scale",
    }
)
#: The kernel switches compared: production, the centred base scheme, and every other switch of `KernelSettings`.
SETTINGS = {
    "production": PRODUCTION_KERNELS,
    "centred": CENTRED_SCHEME,
    "minmod": KernelSettings(density_limiter=DensityLimiter.MINMOD),
    "averaged_uncapped": KernelSettings(viscous_flux=ViscousFlux.AVERAGED, cap_tension=False),
}


# --- the comparison ---


def entries(obj: object, prefix: str = "") -> dict[str, FloatArray | None]:
    """Every array and scalar inside a (nested) dataclass, by its dotted name; `None` for an absent kernel result."""
    if obj is None:
        return {prefix: None}
    if isinstance(obj, np.ndarray):
        return {prefix: cast(FloatArray, obj)}
    if isinstance(obj, (float, int)):
        return {prefix: np.array([float(obj)])}
    assert is_dataclass(obj), prefix
    assert not isinstance(obj, type), prefix
    out: dict[str, FloatArray | None] = {}
    for f in fields(obj):
        out |= entries(getattr(obj, f.name), f"{prefix}.{f.name}" if prefix else f.name)
    return out


def ulps(a: float, b: float) -> int:
    """The distance of two finite doubles in units in the last place."""
    ia, ib = (int(np.array([x]).view(np.int64)[0]) for x in (a, b))
    ia, ib = (i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF) for i in (ia, ib))
    return abs(ia - ib)


type Allowance = dict[str, FloatArray]


def assert_same_result(a: DerivsResult, b: DerivsResult, label: str, allowance: Allowance | None = None) -> None:
    """The two results agree entry by entry: the same NaN entries, and equal values, or, with an `allowance` (the
    general-w comparison), values within it entry by entry. A failure names the field, the first differing entry and
    the distance in ulp."""
    ea, eb = entries(a), entries(b)
    assert ea.keys() == eb.keys(), label
    for name, x in ea.items():
        y = eb[name]
        if x is None or y is None:
            assert x is None, f"{label}: {name} is {x!r} against None"
            assert y is None, f"{label}: {name} is None against {y!r}"
            continue
        assert x.shape == y.shape, f"{label}: {name} has shape {x.shape} against {y.shape}"
        nan_x, nan_y = np.isnan(x), np.isnan(y)
        assert np.array_equal(nan_x, nan_y), f"{label}: {name} is NaN at {np.flatnonzero(nan_x != nan_y)[:5]}"
        values = ~nan_x
        if allowance is None or name in LAPSE_FREE:
            differ = np.flatnonzero(values & (x != y))
        else:
            with np.errstate(invalid="ignore"):  # inf - inf, where the allowance is 0 and equality is what counts
                off = np.where(x == y, 0.0, np.abs(x - y))
            differ = np.flatnonzero(values & ~(off <= allowance[name]))
        if differ.size:
            i = int(differ[0])
            raise AssertionError(
                f"{label}: {name}[{i}] = {x[i]!r} against {y[i]!r}, {ulps(x[i], y[i])} ulp ({differ.size} entries)"
            )


#: The four sign patterns of the lapse error, as (pattern of `e^phi`, pattern of `e^phi - 1`) at alternate entries.
LAPSE_PATTERNS = (
    ((1.0, 1.0), (1.0, 1.0)),
    ((-1.0, -1.0), (1.0, 1.0)),
    ((1.0, -1.0), (-1.0, 1.0)),
    ((-1.0, 1.0), (1.0, -1.0)),
)


def with_lapse_off(
    signs: tuple[tuple[float, float], tuple[float, float]],
) -> Callable[..., tuple[FloatArray, FloatArray]]:
    """`EquationOfState.lapse_and_deviation` with both results moved by `LAPSE_ULPS` ulp, alternate entries by the
    two signs of `signs[0]` (`e^phi`) and `signs[1]` (`e^phi - 1`)."""
    exact = EquationOfState.lapse_and_deviation

    def off(eos: EquationOfState, rho: FloatArray, delta_rho: FloatArray) -> tuple[FloatArray, FloatArray]:
        ephi, delta_ephi = exact(eos, rho, delta_rho)
        n = np.arange(ephi.size)
        s_e = np.where(n % 2 == 0, signs[0][0], signs[0][1])
        s_d = np.where(n % 2 == 0, signs[1][0], signs[1][1])
        return ephi * (1.0 + LAPSE_ULPS * EPS * s_e), delta_ephi * (1.0 + LAPSE_ULPS * EPS * s_d)

    return off


def lapse_allowance(sch: Scheme, xi: float, packed: FloatArray, deviation: bool, a: DerivsResult) -> Allowance:
    """What a lapse error of `LAPSE_ULPS` ulp can change in each entry of the numpy result `a`: `LAPSE_MARGIN` times
    the largest response to the four sign patterns, plus `LAPSE_ULPS` ulp of the entry itself. The fields of
    `LAPSE_FREE` must not respond at all, which checks that list."""
    responses: dict[str, FloatArray] = {}
    base = entries(a)
    for signs in LAPSE_PATTERNS:
        with patch.object(EquationOfState, "lapse_and_deviation", with_lapse_off(signs)):
            moved = outcome(sch, xi, packed, deviation)
        assert isinstance(moved, DerivsResult), f"a lapse error of {LAPSE_ULPS} ulp gave {moved!r}"
        for name, z in entries(moved).items():
            x = base[name]
            if x is None or z is None:
                continue
            with np.errstate(invalid="ignore"):
                r = np.where((x == z) | (np.isnan(x) & np.isnan(z)), 0.0, np.abs(z - x))
            r[np.isnan(r)] = np.inf  # an entry that becomes NaN, or stops being NaN, has no bound
            responses[name] = np.maximum(responses.get(name, 0.0), r)
    allowance: Allowance = {}
    for name, r in responses.items():
        if name in LAPSE_FREE:
            assert not np.any(r), f"{name} responds to the lapse"
        x = base[name]
        assert x is not None
        with np.errstate(invalid="ignore"):
            allowance[name] = LAPSE_MARGIN * r + LAPSE_ULPS * EPS * np.abs(x)
    return allowance


type Outcome = DerivsResult | NotHyperbolicError | ValueError


def outcome(sch: Scheme, xi: float, y: FloatArray, deviation: bool) -> Outcome:
    """The stage at the packed deviation (`evaluate_deviation`) or the packed state (`evaluate`), or its refusal."""
    try:
        return sch.evaluate_deviation(xi, y) if deviation else sch.evaluate(xi, y)
    except (NotHyperbolicError, ValueError) as e:
        return e


def assert_same_outcome(a: Outcome, b: Outcome, label: str, allowance: Allowance | None = None) -> None:
    """The same result, or the same refusal: exception class, message, and for `NotHyperbolicError` the same field,
    index and value (bit for bit, or both NaN)."""
    if isinstance(a, DerivsResult) and isinstance(b, DerivsResult):
        assert_same_result(a, b, label, allowance)
        return
    assert type(a) is type(b), f"{label}: {a!r} against {b!r}"
    assert str(a) == str(b), label
    if isinstance(a, NotHyperbolicError):
        assert isinstance(b, NotHyperbolicError)
        assert (a.field, a.index) == (b.field, b.index), label
        assert np.float64(a.value).tobytes() == np.float64(b.value).tobytes() or (
            math.isnan(a.value) and math.isnan(b.value)
        ), label


@dataclass(frozen=True)
class Pair:
    """The same Scheme on the two engines."""

    numpy: Scheme
    rust: Scheme

    @classmethod
    def of(
        cls,
        eos: EquationOfState,
        m: Map,
        layout: Layout,
        outer: OuterClosure,
        settings: KernelSettings,
        spacetime: Spacetime = Spacetime.FRW,
    ) -> Pair:
        return cls(
            Scheme(eos, m, layout, outer, settings, spacetime),
            Scheme(eos, m, layout, outer, settings, spacetime, Engine.RUST),
        )

    def assert_agree(self, xi: float, y: FloatArray, label: str, exact: bool = True) -> None:
        """Both entry points agree: `evaluate` at the packed state and `evaluate_deviation` at its deviation; exactly,
        or (`exact=False`, at a general `w`) within the lapse allowance of each entry."""
        dy = y - self.numpy.frw(xi)
        for deviation, packed in ((True, dy), (False, y)):
            a = outcome(self.numpy, xi, packed, deviation)
            b = outcome(self.rust, xi, packed, deviation)
            allowance = None
            if not exact and isinstance(a, DerivsResult):
                allowance = lapse_allowance(self.numpy, xi, packed, deviation, a)
            assert_same_outcome(a, b, f"{label} ({'deviation' if deviation else 'state'})", allowance)


def case_pair(case: Case, settings: KernelSettings, eos: EquationOfState | None = None) -> Pair:
    """The case's Scheme on both engines: the outgoing-wave closure, or the held face on the pinned map."""
    outer = OutgoingWave() if case.closure == "sat" else HeldAtFrw()
    return Pair.of(case.eos if eos is None else eos, case.map, case.layout, outer, settings)


#: Penalty strengths other than production's (2, 1, 0), inside the admissible range: with `tau_W = 0` the `W` term of
#: the outgoing-wave rows vanishes identically, and with the production values a slip between the other two would too.
ODD_STRENGTHS = PenaltyStrengths(tau_u=1.3, tau_rho=0.9, tau_W=0.7)
#: Every closure with Rust rows, the outgoing-wave one at production and at odd strengths.
CLOSURES: dict[str, OuterClosure] = {
    "sat": OutgoingWave(),
    "sat_odd": OutgoingWave(ODD_STRENGTHS),
    "held": HeldAtFrw(),
    "exterior": HeldExterior(rho_N=1.7, ephi_N=0.8),
}


# --- the stage on violent states ---


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("w", list(W_CASES))
def test_the_engines_agree_on_violent_states_with_every_kernel_switch(w: str, family: str):
    for seed in range(3):
        case = violent_case(family, seed, w)
        for name, settings in SETTINGS.items():
            pair = case_pair(case, settings)
            pair.assert_agree(case.xi, pair.numpy.layout.pack(case.state), f"{w} {family} {seed} {name}")


@pytest.mark.slow
@pytest.mark.parametrize("family", FAMILIES)
def test_the_engines_agree_on_every_seed_and_at_a_general_w(family: str):
    general = EquationOfState(Fraction(1, 5))  # the lapse by pow, log1p and expm1
    for seed in SEEDS:
        for w in W_CASES:
            case = violent_case(family, seed, w)
            for name, settings in SETTINGS.items():
                pair = case_pair(case, settings)
                pair.assert_agree(case.xi, pair.numpy.layout.pack(case.state), f"{w} {family} {seed} {name}")
        case = violent_case(family, seed)
        for name, settings in SETTINGS.items():
            pair = case_pair(case, settings, general)
            y = pair.numpy.layout.pack(case.state)
            pair.assert_agree(case.xi, y, f"w=1/5 {family} {seed} {name}", exact=False)


@pytest.mark.parametrize("w", list(W_CASES))
def test_the_engines_agree_on_violent_states_with_odd_penalty_strengths(w: str):
    closure = OutgoingWave(ODD_STRENGTHS)
    for family in FAMILIES:
        for seed in range(3):
            case = violent_case(family, seed, w)
            if case.closure != "sat":
                continue  # the pinned family moves its outer face, which the outgoing-wave closure refuses
            case = replace(case, state=replace(case.state, W=0.3 * (seed - 1)))  # an incoming amplitude to penalise
            for name in ("production", "centred"):
                pair = Pair.of(case.eos, case.map, case.layout, closure, SETTINGS[name])
                pair.assert_agree(case.xi, pair.numpy.layout.pack(case.state), f"odd {w} {family} {seed} {name}")


def test_the_engines_agree_on_the_smallest_grids_and_at_the_last_excision_faces():
    # N = 2 to 7, with j_e = 0, 1 and N - 2 (two retained cells, the fewest a Layout allows): the index edges of
    # every Rust loop, under every
    # kernel switch and closure, in FRW and flat spacetime (where only the held face closes: the others refuse, as
    # `test_the_closures_refuse_...` checks). The states are FRW with the density up to 60 per cent higher or three
    # times lower and the velocity 30 per cent off; most give a result and the rest a trapped face. numpy may warn at
    # a trapped face, so the comparison is with its warnings silenced (see `test_numpy_warns_...`).
    rng = np.random.default_rng(11)
    for w, eos in (("w=1/3", RAD), ("w=1", EquationOfState(Fraction(1)))):
        for N in (2, 3, 4, 7):
            for j_e in sorted({j for j in (0, 1, N - 2) if j <= N - 2}):
                for m in (IdentityMap(3.0), SinhStretch(5.0, 1.5)):
                    for spacetime in (Spacetime.FRW, Spacetime.FLAT) if j_e == 0 else (Spacetime.FRW,):
                        for name, settings in SETTINGS.items():
                            for closure_name, closure in CLOSURES.items():
                                if spacetime is Spacetime.FLAT and closure_name != "held":
                                    continue
                                layout = Layout(N, j_e)
                                pair = Pair.of(eos, m, layout, closure, settings, spacetime)
                                xi = float(rng.uniform(0.0, 2.0))
                                frame = pair.numpy.frame(xi)
                                frw = frame.reference.state
                                state = State(
                                    E=frw.E * rng.uniform(0.3, 1.6, N),
                                    U=frw.U + frame.geo.X * rng.uniform(-0.3, 0.3, N + 1),
                                    W=float(rng.uniform(-0.5, 0.5)),
                                    M_e=frw.M_e * float(rng.uniform(0.8, 1.2)),
                                )
                                label = f"{w} N={N} j_e={j_e} {m!r} {spacetime.name} {name} {closure_name}"
                                with np.errstate(all="ignore"):
                                    pair.assert_agree(xi, layout.pack(state), label)


def refused_state(case: Case, what: str) -> State:
    """The case with a negative density, an outer face density that rounds to zero, a trapped interior
    (`Gammabar^2 < 0`), or a non-finite energy."""
    E, U = case.state.E.copy(), case.state.U.copy()
    if what == "rho":
        E[7] = -E[7]
    elif what == "rho_N":
        E[-1] *= 1e-17  # positive, but the deviation form's face density is `1 + delta`, which rounds to zero
    elif what == "Gammabar2":
        E[:20] *= 1e3  # far more mass inside than the expansion and the velocity can hold
        U[1:21] = 0.0
    else:
        E[5] = math.nan
    return State(E=E, U=U, W=case.state.W, M_e=case.state.M_e)


@pytest.mark.parametrize("what", ["rho", "rho_N", "Gammabar2", "nonfinite"])
def test_a_state_outside_the_hyperbolic_domain_is_refused_alike(what: str):
    case = violent_case("stretched", 0)
    state = refused_state(case, what)
    for name, settings in SETTINGS.items():
        pair = case_pair(case, settings)
        y = pair.numpy.layout.pack(state)
        with pytest.raises(NotHyperbolicError) as refused:
            pair.numpy.evaluate(case.xi, y)
        assert refused.value.field == ("Gammabar2" if what == "Gammabar2" else "rho")
        assert (refused.value.index == case.layout.N) == (what == "rho_N")
        assert math.isfinite(refused.value.value) == (what != "nonfinite")
        pair.assert_agree(case.xi, y, f"{what} {name}")


def test_a_vector_that_is_not_native_float64_is_taken_or_refused_alike():
    case = violent_case("shock", 0)
    pair = case_pair(case, PRODUCTION_KERNELS)
    y = pair.numpy.layout.pack(case.state)
    dy = y - pair.numpy.frw(case.xi)
    # other dtypes, converted exactly by the numpy engine's `unpack` and by the Rust adapter alike, and a 2-D vector,
    # refused by the same check with the same message
    for deviation, packed in ((True, dy), (False, y)):
        for other in (packed.astype(">f8"), packed.astype(np.float32), np.rint(packed).astype(np.int64), packed[None]):
            a = outcome(pair.numpy, case.xi, cast(FloatArray, other), deviation)
            b = outcome(pair.rust, case.xi, cast(FloatArray, other), deviation)
            assert_same_outcome(a, b, f"{other.dtype} {other.shape}")
            assert isinstance(a, ValueError) == (other.ndim == 2)  # the rounded states may well be refused


def test_numpy_warns_where_the_rust_engine_is_silent_and_otherwise_refuses_alike():
    # A deviation that overflows the cumulative mass: numpy emits a RuntimeWarning, which the suite's
    # `filterwarnings = ["error"]` turns into an exception; the Rust engine has no warnings, and goes on to the refusal
    # that numpy reaches with its warnings silenced.
    pair = Pair.of(RAD, IdentityMap(2.0), Layout(8), OutgoingWave(), PRODUCTION_KERNELS)
    dy = np.zeros(pair.numpy.layout.size)
    dy[:8] = pair.numpy.frame(0.0).geo.dV * 1e308
    with pytest.raises(RuntimeWarning, match="overflow"):
        pair.numpy.evaluate_deviation(0.0, dy)
    with np.errstate(all="ignore"):
        a = outcome(pair.numpy, 0.0, dy, True)
    b = outcome(pair.rust, 0.0, dy, True)
    assert isinstance(b, NotHyperbolicError)
    assert_same_outcome(a, b, "overflow")


@pytest.mark.parametrize("engine", list(Engine))
def test_the_closures_refuse_what_they_cannot_close_on_either_engine(engine: Engine):
    pinned = PinnedMap(IdentityMap(4.0), float(RAD.alpha))
    sch = Scheme(RAD, pinned, Layout(8), OutgoingWave(), CENTRED_SCHEME, engine=engine)
    with pytest.raises(ValueError, match="static outer face"):
        sch.evaluate(0.3, sch.frw(0.3))
    flat = Scheme(RAD, IdentityMap(4.0), Layout(8), OutgoingWave(), PRODUCTION_KERNELS, Spacetime.FLAT, engine)
    with pytest.raises(ValueError, match="derived about FRW"):
        flat.evaluate_deviation(0.0, np.zeros(flat.layout.size))
    held = HeldExterior(rho_N=1.0, ephi_N=1.0)
    flat = Scheme(RAD, IdentityMap(4.0), Layout(8), held, PRODUCTION_KERNELS, Spacetime.FLAT, engine)
    with pytest.raises(ValueError, match="steady flow about a hole"):
        flat.evaluate(0.0, flat.frw(0.0))


def test_the_engines_agree_on_the_michel_flow_on_the_pinned_map():
    eps, xi_0, dX = 1e-8, 0.0, 0.05
    N, X_max = michel_grid(eps, xi_0, RAD, 5.0, dX)
    layout = Layout(N, j_e=round(1.5 / dX))  # excised at 1.5 M
    flow = michel_flow(np.array([5.0]))
    held = HeldExterior(rho_N=float(flow.compression[0]), ephi_N=float(flow.N[0]))
    pinned = PinnedMap(IdentityMap(X_max), float(RAD.alpha), xi_on=xi_0)
    for name, settings in SETTINGS.items():
        pair = Pair.of(RAD, pinned, layout, held, settings)
        for xi in (xi_0, xi_0 + 0.1, xi_0 + 0.37):
            frame = pair.numpy.frame(xi)
            state = michel_state(frame.geo, frame.bg, RAD, eps, layout)
            pair.assert_agree(xi, layout.pack(state), f"michel {name} {xi}")


def test_the_engines_agree_on_the_flat_tube():
    h, N = 0.006, 1000
    rng = np.random.default_rng(6)
    for name, settings in SETTINGS.items():
        pair = Pair.of(RAD, IdentityMap(N * h), Layout(N), HeldAtFrw(), settings, Spacetime.FLAT)
        geo = pair.numpy.frame(0.0).geo
        rho = np.where(np.arange(N) < 167, 4.013, 1.0) * rng.uniform(0.5, 1.5, N)  # the compression-2 shock tube
        U = np.concatenate(([0.0], rng.uniform(-0.3, 0.3, N)))
        pair.assert_agree(0.0, pair.numpy.layout.pack(State(E=rho * geo.dV, U=U, W=0.0)), f"flat {name}")


# --- the checked step on both engines ---


def assert_same_step(a: AcceptedStep, b: AcceptedStep, label: str) -> None:
    """Two accepted steps agree: the step, the arrival, every stage and the result, and the refusals."""
    assert a.dxi == b.dxi, label
    assert np.array_equal(a.dy, b.dy, equal_nan=True), label
    assert_same_result(a.result, b.result, f"{label} result")
    assert len(a.stages) == len(b.stages), label
    for n, (sa, sb) in enumerate(zip(a.stages, b.stages, strict=True)):
        assert (sa.xi, sa.fluxes) == (sb.xi, sb.fluxes), f"{label} stage {n}"
        assert np.array_equal(sa.k, sb.k, equal_nan=True), f"{label} stage {n}"
    assert a.refused == b.refused, label


def test_a_step_on_the_rust_engine_builds_no_python_frame_inside_the_step():
    # the stages at xi + dxi / 2 read only the Rust frame; the step's arrival is a whole frame, which the driver reads
    zone = Zone(xi_on=0.4, tau_on=0.3, x_t=0.3, Delta_t=0.1)
    m = BlendMap(SinhStretch(6.0, scale=2.0), 0.5, (zone,))
    sch = Scheme(RAD, m, Layout(40), OutgoingWave(), PRODUCTION_KERNELS, engine=Engine.RUST)
    xi, dxi = 0.5, 0.02
    dy = blob(sch, xi)
    accepted = advance_checked(sch, xi, dy, dxi, sch.evaluate_deviation(xi, dy))
    assert accepted.refused == []
    held = sch._frames  # pyright: ignore[reportPrivateUsage]
    inside = sch._stage_frames  # pyright: ignore[reportPrivateUsage]
    assert set(held) == {xi, xi + dxi}
    assert set(inside) == {xi + 0.5 * dxi}
    for n in range(8):  # more step times than are kept: the oldest go
        t = xi + dxi + 0.01 * n
        advance_checked(sch, t, accepted.dy, 0.005, sch.evaluate_deviation(t, accepted.dy))
    assert len(inside) == FRAMES_KEPT


def test_a_checked_attempt_refuses_alike_on_both_engines():
    # steps far too long for the violent states, so that the attempts are refused at a stage input or at the result,
    # for density, Gammabar^2 or a non-finite value: the same refusal (cause, stage, index, and the value bit for bit
    # or both NaN), after the same stages; and where an attempt passes, the same arrival
    causes: set[str] = set()
    for family in FAMILIES:
        for seed in SEEDS[:6]:
            case = violent_case(family, seed)
            pair = case_pair(case, PRODUCTION_KERNELS)
            y = pair.numpy.layout.pack(case.state)
            dy = y - pair.numpy.frw(case.xi)
            try:
                firsts = [sch.evaluate_deviation(case.xi, dy) for sch in (pair.numpy, pair.rust)]
            except NotHyperbolicError, ValueError:
                continue
            base = step_size(firsts[0], pair.numpy.frame(case.xi).geo, case.layout, 0.75, 1.0).dxi
            for factor in (1.0, 8.0, 64.0, 512.0):
                label = f"{family} {seed} x{factor}"
                a, b = (
                    checked_step(sch, case.xi, dy, factor * base, f)
                    for sch, f in zip((pair.numpy, pair.rust), firsts, strict=True)
                )
                assert (a.failure is None) == (b.failure is None), label
                if a.failure is not None and b.failure is not None:
                    fa, fb = a.failure, b.failure
                    assert (fa.cause, fa.stage, fa.index) == (fb.cause, fb.stage, fb.index), label
                    assert np.array_equal([fa.value], [fb.value], equal_nan=True), label
                    causes.add(fa.cause.value)
                else:
                    assert a.dy is not None, label
                    assert b.dy is not None, label
                    assert np.array_equal(a.dy, b.dy, equal_nan=True), label
                    assert a.result is not None, label
                    assert b.result is not None, label
                    assert_same_result(a.result, b.result, label)
                assert len(a.stages) == len(b.stages), label
                for sa, sb in zip(a.stages, b.stages, strict=True):
                    assert (sa.xi, sa.fluxes) == (sb.xi, sb.fluxes), label
                    assert np.array_equal(sa.k, sb.k, equal_nan=True), label
    assert len(causes) >= 3, causes  # refusals of more than one kind were compared


def march(pair: Pair, xi: float, dy: FloatArray, steps: int, cap: float, label: str) -> None:
    """Checked RK4 steps on both engines from the same deviation, each engine choosing its own step."""
    first = [sch.evaluate_deviation(xi, dy) for sch in (pair.numpy, pair.rust)]
    assert_same_result(first[0], first[1], f"{label} first")
    dys = [dy, dy]
    for n in range(steps):
        accepted: list[AcceptedStep] = []
        for k, sch in enumerate((pair.numpy, pair.rust)):
            dxi = step_size(first[k], sch.frame(xi).geo, sch.layout, 0.75, cap).dxi
            accepted.append(advance_checked(sch, xi, dys[k], dxi, first[k]))
        assert_same_step(accepted[0], accepted[1], f"{label} step {n}")
        first = [a.result for a in accepted]
        dys = [a.dy for a in accepted]
        xi += accepted[0].dxi


def blob(sch: Scheme, xi: float) -> FloatArray:
    """A strong smooth compression at the centre, as a deviation."""
    geo = sch.frame(xi).geo
    E = geo.dV * (1.0 + 0.5 * np.exp(-((geo.Xm / 1.5) ** 2)))
    U = geo.X * (1.0 - 0.1 * np.exp(-((geo.X / 1.5) ** 2)))
    return sch.layout.pack(State(E=E, U=U, W=0.0)) - sch.frw(xi)


def test_a_short_checked_march_on_a_moving_map_is_the_same_on_both_engines():
    zone = Zone(xi_on=0.4, tau_on=0.3, x_t=0.3, Delta_t=0.1)
    m = BlendMap(SinhStretch(6.0, scale=2.0), 0.5, (zone,))
    for name, settings in SETTINGS.items():
        pair = Pair.of(RAD, m, Layout(60), OutgoingWave(), settings)
        march(pair, 0.45, blob(pair.numpy, 0.45), 5, step_cap(RAD), f"blend {name}")


@pytest.mark.slow
@pytest.mark.parametrize("settings", [PRODUCTION_KERNELS, CENTRED_SCHEME])
@pytest.mark.parametrize("outer", [OutgoingWave(), HeldAtFrw()])
def test_sixty_checked_steps_are_the_same_on_both_engines(outer: OuterClosure, settings: KernelSettings):
    pair = Pair.of(RAD, SinhStretch(12.0, scale=3.0), Layout(300), outer, settings)
    march(pair, 3.0, blob(pair.numpy, 3.0), 60, step_cap(RAD), f"{type(outer).__name__} {settings.kernels.value}")


@pytest.mark.slow
def test_a_collapse_runs_to_its_read_out_through_the_same_events_on_both_engines(tmp_path: Path):
    readers: list[RunReader] = []
    A = 0.515 * math.e / 8.0
    path = tmp_path / "collapse.yaml"
    path.write_text(
        "grid: {N: 100, Rtilde_max: 12.0, scale: 3.0}\noutput: {snapshots: milestones}\nevolution: {xi_end: 9.0}\n"
    )
    for engine in Engine:
        args = ["initial", "gaussian", engine.value, "--config", str(path), "--A", f"{A:.12g}", "--ell", "2.0"]
        assert main([*args, "--dir", str(tmp_path)]) == 0
        assert main(["run", str(path), engine.value, "--engine", engine.value, "--dir", str(tmp_path)]) == 0
        paths = RunPaths.of(tmp_path, engine.value)
        readers.append(RunReader(paths.evolution))
        assert yaml.safe_load(paths.config.read_text())["provenance"]["engine"] == engine.value
    a, b = readers
    kinds = [e.kind for e in a.events]
    assert {"formation", "switch_on", "re_excision", "readout"} <= set(kinds)
    assert a.events == b.events  # the same events at the same steps and times, with the same payloads (M_est)
    for table in ("steps", "horizon"):
        ta, tb = getattr(a, table), getattr(b, table)
        assert ta.keys() == tb.keys()
        for name, column in ta.items():
            if isinstance(column, list):  # a column of strings
                assert column == tb[name], (table, name)
            else:
                assert np.array_equal(column, np.asarray(tb[name]), equal_nan=True), (table, name)
    assert [s.step for s in a.snapshots] == [s.step for s in b.snapshots]
    for s in a.snapshots:
        ra, rb = a.snapshot(s.index), b.snapshot(s.index)
        assert np.array_equal(ra.delta_E, rb.delta_E, equal_nan=True)
        assert np.array_equal(ra.delta_U, rb.delta_U, equal_nan=True)
        assert (ra.W, ra.M_e, ra.xi, ra.j_e, ra.xi_form, ra.zones) == (
            rb.W,
            rb.M_e,
            rb.xi,
            rb.j_e,
            rb.xi_form,
            rb.zones,
        )
    assert a.config == b.config
    assert (a.engine, b.engine) == (Engine.PYTHON, Engine.RUST)


# --- the horizon finder on both engines ---


def assert_same_trapping(a: Trapping, b: Trapping, label: str) -> None:
    """The two engines' finder numbers agree: `h` entry by entry (NaN in the same entries), the rest exactly."""
    assert np.array_equal(a.h, b.h, equal_nan=True), label
    assert a.crossings == b.crossings, label
    assert (a.trapped_faces, a.margin, a.margin_face, a.core_margin, a.core_margin_face, a.outer_face_trapped) == (
        b.trapped_faces,
        b.margin,
        b.margin_face,
        b.core_margin,
        b.core_margin_face,
        b.outer_face_trapped,
    ), label
    assert all(type(j) is int and type(t) is float and type(o) is bool for j, t, o in b.crossings), label
    assert (type(b.trapped_faces), type(b.margin), type(b.margin_face)) == (int, float, int), label


def test_the_finder_computes_the_same_numbers_on_both_engines():
    # trapping functions that change sign often, on grids from the smallest a layout allows, before and after
    # excision: every root with its cubic, the linear roots beside the first retained and the outer face, the core
    # ending at the first retained face or running to the outer face, a root exactly at a face (h = 0), and the
    # central infall region of every length
    rng = np.random.default_rng(23)
    for N in (2, 3, 4, 7, 40, 400):
        for j_e in sorted({0, 1, N // 3, N - 2} & set(range(N - 1))):
            for trial in range(12):
                layout = Layout(N, j_e)
                Gammabar2 = rng.uniform(0.05, 4.0, N + 1)
                wiggle = rng.uniform(-1.0, 1.0, N + 1) * rng.choice([0.02, 0.3, 2.0])
                U = -np.sqrt(Gammabar2) * (1.0 + wiggle)
                if trial % 4 == 1:
                    U[j_e + 1 :] = np.abs(U[j_e + 1 :])  # the centre falls in, everything outside expands
                elif trial % 4 == 2:
                    U = -np.abs(U)  # the whole grid falls in
                elif trial % 4 == 3 and N > 3:
                    U[j_e + 1] = -np.sqrt(Gammabar2[j_e + 1])  # a root exactly on a face
                U[:j_e] = np.nan
                Gammabar2[:j_e] = np.nan
                label = f"N={N} j_e={j_e} trial={trial}"
                assert_same_trapping(trapping(U, Gammabar2, layout), rust_engine.trapping(U, Gammabar2, layout), label)


def test_the_finder_reports_alike_on_violent_states_and_on_a_collapse():
    # the whole report, with the radii from the map, on the violent states (the trapped and excised ones among them)
    # and on the blob of a strong compression on a moving map
    for family in FAMILIES:
        for seed in SEEDS:
            case = violent_case(family, seed)
            pair = case_pair(case, PRODUCTION_KERNELS)
            y = pair.numpy.layout.pack(case.state)
            try:
                result = pair.numpy.evaluate(case.xi, y)
            except NotHyperbolicError, ValueError:
                continue
            frame = pair.numpy.frame(case.xi)
            reports = [
                find_horizons(
                    case.state, result.derived, frame.geo, frame.bg, case.eos, case.map, case.layout, case.xi, e
                )
                for e in Engine
            ]
            a, b = reports
            label = f"{family} {seed}"
            assert np.array_equal(a.h, b.h, equal_nan=True), label
            assert (a.horizons, a.apparent, a.trapped_faces) == (b.horizons, b.apparent, b.trapped_faces), label
            assert (a.margin, a.margin_face, a.core_margin, a.core_margin_face) == (
                b.margin,
                b.margin_face,
                b.core_margin,
                b.core_margin_face,
            ), label
            assert np.array_equal([a.M_AH, a.residual], [b.M_AH, b.residual], equal_nan=True), label
            assert a.outer_face_trapped == b.outer_face_trapped, label


def test_the_rust_finder_refuses_what_it_cannot_read():
    U, Gammabar2 = np.full(9, -1.0), np.ones(9)
    for bad_U, bad_G, j_e in ((U, Gammabar2[:-1], 0), (U[:2], Gammabar2[:2], 0), (U, Gammabar2, 7)):
        with pytest.raises(ValueError, match="the finder needs"):
            pbh_engine.trapping(bad_U, bad_G, j_e)
    with pytest.raises(ValueError, match="All-NaN slice encountered"):  # as `np.nanargmin` refuses
        pbh_engine.trapping(np.full(9, np.nan), Gammabar2, 0)


# --- the switch ---


MINIMAL = "grid: {N: 40, Rtilde_max: 8.0, scale: 3.0}\nevolution: {xi_end: 1.0}\n"


def test_the_engine_is_chosen_with_the_scheme_defaults_to_numpy_and_is_recorded_only_as_provenance(tmp_path: Path):
    path = tmp_path / "minimal.yaml"
    path.write_text(MINIMAL)
    config = load(path)
    sch = config.scheme()
    assert sch.engine is Engine.PYTHON
    assert sch.frame(0.0).rust is None
    rust = config.scheme(engine=Engine.RUST)
    assert rust.engine is Engine.RUST
    assert rust.frame(0.0).rust is not None
    save(config, tmp_path / "rust.yaml", Engine.RUST)  # a run's configuration, as the driver saves it
    assert yaml.safe_load((tmp_path / "rust.yaml").read_text())["provenance"]["engine"] == "rust"
    assert load(tmp_path / "rust.yaml") == config  # provenance, not configuration: it does not come back
    save(config, tmp_path / "plain.yaml")  # anything else saved has no engine
    assert "engine" not in yaml.safe_load((tmp_path / "plain.yaml").read_text())["provenance"]
    path.write_text(MINIMAL + "numerics: {engine: rust}\n")
    with pytest.raises(ConfigError, match="numerics"):  # not a section of the configuration
        load(path)


def test_importing_the_package_and_the_numpy_engine_does_not_load_the_extension():
    code = "import sys, pbh, pbh.timestep, pbh.driver, pbh.config, pbh.cli; print('pbh_engine' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


@dataclass(frozen=True)
class Reflecting(OuterClosure):
    """A closure the Rust engine has no rows for."""

    def rows(self, inputs: OuterInputs, eos: EquationOfState) -> OuterRows:
        return OuterRows(delta_dU_N=0.0, delta_F_N=0.0, dW=0.0)


@dataclass(frozen=True)
class Stronger(OutgoingWave):
    """A subclass of a closure that has rows: it must not silently run its parent's."""


@pytest.mark.parametrize("outer", [Reflecting(), Stronger()])
def test_a_closure_without_rust_rows_is_refused_when_the_scheme_is_built(outer: OuterClosure):
    Scheme(RAD, IdentityMap(4.0), Layout(8), outer, CENTRED_SCHEME)  # the numpy engine runs any closure
    with pytest.raises(ValueError, match="the rust engine has no rows"):
        Scheme(RAD, IdentityMap(4.0), Layout(8), outer, CENTRED_SCHEME, engine=Engine.RUST)


def test_a_static_map_shares_one_rust_frame_and_a_moving_map_builds_one_per_time():
    static = Scheme(
        RAD, SinhStretch(6.0, scale=2.0), Layout(20), OutgoingWave(), PRODUCTION_KERNELS, engine=Engine.RUST
    )
    assert static.frame(0.1).rust is not None
    assert static.frame(0.1).rust is static.frame(0.2).rust
    pinned = PinnedMap(IdentityMap(4.0), float(RAD.alpha))
    moving = Scheme(RAD, pinned, Layout(20), HeldAtFrw(), PRODUCTION_KERNELS, engine=Engine.RUST)
    assert moving.frame(0.1).rust is not moving.frame(0.2).rust
    assert moving.frame(0.1).rust is moving.frame(0.1).rust  # kept with its frame


def test_no_array_of_a_rust_result_is_shared_with_another_result_or_a_frame():
    zone = Zone(xi_on=0.4, tau_on=0.3, x_t=0.3, Delta_t=0.1)
    sch = Scheme(
        RAD,
        BlendMap(SinhStretch(6.0, scale=2.0), 0.5, (zone,)),
        Layout(30),
        OutgoingWave(),
        PRODUCTION_KERNELS,
        engine=Engine.RUST,
    )
    dy = blob(sch, 0.5)
    results = [sch.evaluate_deviation(0.5, dy), sch.evaluate_deviation(0.5, dy), sch.evaluate(0.55, sch.frw(0.55))]
    frame = sch.frame(0.5)
    held = [a for a in entries(frame.geo).values() if a is not None] + [dy]
    for n, r in enumerate(results):
        arrays = [a for a in entries(r).values() if a is not None and a.size > 1]
        others = held + [a for m, o in enumerate(results) if m != n for a in entries(o).values() if a is not None]
        assert not any(np.shares_memory(a, b) for a in arrays for b in others)


@pytest.mark.parametrize("engine", list(Engine))
def test_a_scheme_pickles_and_deep_copies_on_either_engine_and_evaluates_alike(engine: Engine):
    zone = Zone(xi_on=0.4, tau_on=0.3, x_t=0.3, Delta_t=0.1)
    blend = BlendMap(SinhStretch(6.0, scale=2.0), 0.5, (zone,))
    sch = Scheme(RAD, blend, Layout(30), OutgoingWave(), PRODUCTION_KERNELS, engine=engine)
    dy = blob(sch, 0.5)
    before = sch.evaluate_deviation(0.5, dy)  # the frame at 0.5 is now in the memo, and is rebuilt by each copy
    for other in (pickle.loads(pickle.dumps(sch)), copy.deepcopy(sch), copy.copy(sch)):
        assert other == sch
        assert other.engine is engine
        assert_same_result(other.evaluate_deviation(0.5, dy), before, f"{engine} copy")


def frame_entries(f: Frame) -> dict[str, object]:
    """Every number a frame holds: the geometry, the stencil weights, the FRW reference and the packed FRW state."""
    ref = f.reference
    return {
        **{f"geo.{k}": getattr(f.geo, k) for k in (field.name for field in fields(f.geo))},
        **{f"w.{k}": getattr(f.w, k) for k in (field.name for field in fields(f.w)) if k != "layout"},
        "state.E": ref.state.E,
        "state.U": ref.state.U,
        "state.M_e": ref.state.M_e,
        "rate.E": ref.rate.E,
        "rate.U": ref.rate.U,
        "rate.M_e": ref.rate.M_e,
        "frw_speed": ref.frw_speed,
        "F_frw": ref.F_frw,
        "y_frw": f.y_frw,
    }


def test_the_rust_engine_builds_the_same_frames_as_the_numpy_engine():
    # every map family, moving and static, before and after excision, at every w the engines compare at, in FRW and
    # flat spacetime: the frame Rust builds, read back into the Python's objects, holds the same numbers, NaN where the
    # Python has NaN (the scalar powers of the reference included, which both take by the C library's `pow`)
    zone = Zone(xi_on=0.4, tau_on=0.3, x_t=0.3, Delta_t=0.1)
    maps: list[Map] = [
        IdentityMap(4.0),
        SinhStretch(24.0, 3.0),
        PinnedMap(SinhStretch(8.0, 2.0), float(RAD.alpha), xi_on=0.2),
        BlendMap(SinhStretch(6.0, scale=2.0), float(RAD.alpha), (zone,)),
    ]
    for eos in (RAD, EquationOfState(Fraction(1)), EquationOfState(Fraction(1, 5))):
        for m in maps:
            for N, j_e in ((2, 0), (3, 1), (40, 0), (40, 7), (41, 39), (400, 23)):
                for spacetime in (Spacetime.FRW, Spacetime.FLAT) if j_e == 0 else (Spacetime.FRW,):
                    for xi in (0.0, 0.45, 0.7, 2.5):
                        schemes = [
                            Scheme(eos, m, Layout(N, j_e), HeldAtFrw(), PRODUCTION_KERNELS, spacetime, engine)
                            for engine in Engine
                        ]
                        a, b = (frame_entries(s.frame(xi)) for s in schemes)
                        label = f"{eos.w} {m!r} N={N} j_e={j_e} {spacetime.name} xi={xi}"
                        assert a.keys() == b.keys()
                        for name, x in a.items():
                            y = b[name]
                            if isinstance(x, np.ndarray):
                                x, y = cast(FloatArray, x), cast(FloatArray, y)
                                assert x.shape == y.shape, (label, name)
                                assert np.array_equal(x, y, equal_nan=True), (label, name)
                                assert not y.flags.writeable, (label, name)
                            else:
                                assert x == y, (label, name, x, y)


def test_the_rust_engine_forms_a_blend_maps_radii_as_the_map_does():
    # one to three zones, before, inside and after each ramp, on grids from the smallest to the production size
    zones = (
        Zone(xi_on=1.0, tau_on=0.3, x_t=0.2, Delta_t=0.05),
        Zone(xi_on=1.4, tau_on=0.25, x_t=0.4, Delta_t=0.1),
        Zone(xi_on=2.0, tau_on=0.5, x_t=0.7, Delta_t=0.15),
    )
    for base in (IdentityMap(8.0), SinhStretch(30.0, 3.0)):
        for n in (1, 2, 3):
            m = BlendMap(base, 0.5, zones[:n])
            for N in (2, 40, 1600):
                stage = RustStage(RAD, PRODUCTION_KERNELS, HeldAtFrw(), Layout(N))
                for xi in (0.0, 1.0, 1.1, 1.5, 2.3, 7.0):
                    for x, y in zip(m.radii(xi, N), stage.blend_radii(m, xi), strict=True):
                        assert np.array_equal(x, y), (base, n, N, xi)
    with pytest.raises(ValueError, match="one weight row per zone"):
        pbh_engine.blend_radii(np.ones(5), np.ones((3, 5)), [1.0], [0.0], 0.5)


def test_the_monitors_formed_in_rust_are_the_numpy_monitors():
    # the emptying rates on every violent state (the excised and pinned ones among them) and the near-zone monitors
    # at radii across and beyond the retained grid, on a grid point, and between points: the same numbers
    rng = np.random.default_rng(29)
    compared = 0
    for family in FAMILIES:
        for seed in SEEDS:
            case = violent_case(family, seed)
            pair = case_pair(case, PRODUCTION_KERNELS)
            try:
                result = pair.numpy.evaluate(case.xi, pair.numpy.layout.pack(case.state))
            except NotHyperbolicError, ValueError:
                continue
            geo, layout, eos, state = pair.numpy.frame(case.xi).geo, case.layout, case.eos, case.state
            a = emptying_rates(result, state, geo, eos, layout)
            b = emptying_rates(result, state, geo, eos, layout, Engine.RUST)
            assert np.array_equal(a, b, equal_nan=True), f"{family} {seed}"
            cells, faces = layout.cells, layout.faces
            d = result.derived
            fields = (geo.Xm[cells], geo.X[faces], d.ephi[cells], d.rho[cells], state.U[faces], d.Gammabar2[faces])
            X = geo.X[faces]
            radii = [
                *rng.uniform(-0.1, 1.1, 6) * float(X[-1]),
                float(X[3]),
                float(geo.Xm[layout.j_e + 2]),
                float(X[-1]),
            ]
            va, ka = near_zone_numbers(*fields, radii)
            vb, kb = rust_engine.near_zone(*fields, radii)
            assert ka == kb, f"{family} {seed}"
            assert np.array_equal(va, vb, equal_nan=True), f"{family} {seed}"
            compared += 1
    assert compared > 50


def test_the_near_zone_row_is_the_same_on_both_engines():
    # the whole row, with and without an apparent horizon: the collapse blob and the violent states' trapped ones
    rows = 0
    for family in ("excised", "shock", "void"):
        for seed in SEEDS:
            case = violent_case(family, seed)
            pair = case_pair(case, PRODUCTION_KERNELS)
            try:
                result = pair.numpy.evaluate(case.xi, pair.numpy.layout.pack(case.state))
            except NotHyperbolicError, ValueError:
                continue
            frame = pair.numpy.frame(case.xi)
            report = find_horizons(
                case.state, result.derived, frame.geo, frame.bg, case.eos, case.map, case.layout, case.xi
            )
            a, b = (
                near_zone(case.state, result.derived, frame.geo, report, case.eos, case.layout, case.xi, engine)
                for engine in Engine
            )
            assert np.array_equal(astuple(a), astuple(b), equal_nan=True), f"{family} {seed}"
            rows += report.apparent is not None
    assert rows > 0  # rows with a horizon were compared


def test_an_inconsistent_frame_is_refused_with_value_error_and_never_panics():
    stage = RustStage(RAD, PRODUCTION_KERNELS, HeldAtFrw(), Layout(20, 4))
    X, X_xi = SinhStretch(6.0, scale=2.0).radii(0.3, 20)
    stage.frame(X, X_xi, 1.0)
    for bad_X, bad_X_xi, match in (
        (X[:5], X_xi[:5], "0 <= j_e <= N - 2"),  # N = 3 cells, fewer than the layout's excision face allows
        (X[:3], X_xi[:3], "N \\+ 2 >= 4"),  # a single cell
        (X, X_xi[:-1], "as many velocities"),
    ):
        with pytest.raises(ValueError, match=match):
            stage.frame(bad_X, bad_X_xi, 1.0)
    for bad in (np.concatenate(([1e-3], X[1:])), X[::-1].copy()):  # refused before Rust, as `Geometry.of` refuses
        with pytest.raises(ValueError, match=r"X_0 = 0 exactly|increase strictly"):
            stage.frame(bad, X_xi, 1.0)


def stub_signature(node: ast.FunctionDef) -> str:
    """A stub's parameter list as the extension's `__text_signature__` prints it: no annotations, no `self`."""
    args = copy.deepcopy(node.args)
    args.posonlyargs = []
    args.args = [a for a in args.args if a.arg != "self"]
    for a in args.args + args.kwonlyargs:
        a.annotation = None
    return f"({ast.unparse(args)})"


def test_the_type_stubs_describe_the_compiled_extension():
    stub = ast.parse((Path(__file__).parents[1] / "rust" / "pbh_engine.pyi").read_text())
    # maturin installs the extension as pbh_engine.pbh_engine inside a package that re-exports all of it
    compiled = importlib.import_module("pbh_engine.pbh_engine")
    public = {name for name in dir(compiled) if not name.startswith("_")}
    assert {name for name in dir(pbh_engine) if not name.startswith("_")} == public | {"pbh_engine"}
    stubbed = {node.name for node in stub.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    assert stubbed == public
    for node in stub.body:
        if isinstance(node, ast.FunctionDef):
            assert stub_signature(node) == getattr(compiled, node.name).__text_signature__, node.name
        elif isinstance(node, ast.ClassDef):
            cls = getattr(compiled, node.name)
            members = {m.name: m for m in node.body if isinstance(m, ast.FunctionDef)}
            init = members.pop("__init__", None)
            assert set(members) == {name for name in dir(cls) if not name.startswith("_")}, node.name
            assert (None if init is None else stub_signature(init)) == cls.__text_signature__, node.name


def stub_value_matches(value: object, annotation: str, classes: dict[str, ast.ClassDef]) -> bool:
    """Whether a value the extension returned is of the type its stub declares, recursing into the output classes."""
    if " | " in annotation:
        return any(stub_value_matches(value, option, classes) for option in annotation.split(" | "))
    if annotation == "None":
        return value is None
    if annotation in ("float", "int", "bool"):
        return type(value).__name__ == annotation
    if annotation == "list[tuple[int, float, bool]]":
        entries = cast(list[object], value) if isinstance(value, list) else None
        return entries is not None and all(
            stub_value_matches(entry, "tuple[int, float, bool]", classes) for entry in entries
        )
    if annotation == "str":
        return type(value) is str
    if annotation == "list[float]":
        entries = cast(list[object], value) if isinstance(value, list) else None
        return entries is not None and all(type(entry) is float for entry in entries)
    if annotation.startswith("list[tuple[") or annotation.startswith("tuple["):
        inner = annotation.removeprefix("list[")[: -1 if annotation.startswith("list[") else None]
        parts_types = [part.strip() for part in inner.removeprefix("tuple[").removesuffix("]").split(",")]
        if annotation.startswith("list["):
            entries = cast(list[object], value) if isinstance(value, list) else None
            return entries is not None and all(stub_value_matches(entry, inner, classes) for entry in entries)
        if not isinstance(value, tuple):
            return False
        parts = cast(tuple[object, ...], value)
        return len(parts) == len(parts_types) and all(
            stub_value_matches(v, a, classes) for v, a in zip(parts, parts_types, strict=True)
        )
    if annotation == "tuple[int, float, bool]":
        if not isinstance(value, tuple):
            return False
        parts = cast(tuple[object, ...], value)
        return len(parts) == 3 and all(
            stub_value_matches(v, a, classes) for v, a in zip(parts, ("int", "float", "bool"), strict=True)
        )
    if annotation == "FloatArray":
        if not isinstance(value, np.ndarray):
            return False
        array = cast(FloatArray, value)
        return array.dtype == np.float64 and array.ndim == 1
    node = classes[annotation]
    if type(value).__name__ != annotation:
        return False
    getters = [m for m in node.body if isinstance(m, ast.FunctionDef) and m.returns is not None]
    for getter in getters:
        declared = ast.unparse(cast(ast.expr, getter.returns))
        assert stub_value_matches(getattr(value, getter.name), declared, classes), f"{annotation}.{getter.name}"
    return True


def test_what_the_extension_returns_has_the_types_its_stubs_declare(monkeypatch: pytest.MonkeyPatch):
    from pbh import rust_engine

    stub = ast.parse((Path(__file__).parents[1] / "rust" / "pbh_engine.pyi").read_text())
    classes = {node.name: node for node in stub.body if isinstance(node, ast.ClassDef)}
    returns = {node.name: node.returns for node in stub.body if isinstance(node, ast.FunctionDef)}
    captured: list[pbh_engine.StageOutput] = []
    to_result = rust_engine.to_result

    def capture(out: pbh_engine.StageOutput) -> DerivsResult:
        """Keep the raw output of each stage, and assemble it as the adapter does."""
        captured.append(out)
        return to_result(out)

    monkeypatch.setattr(rust_engine, "to_result", capture)
    for settings in (PRODUCTION_KERNELS, CENTRED_SCHEME):  # with a kernel result, and without one
        for j_e in (0, 4):
            sch = Scheme(RAD, SinhStretch(6.0, scale=2.0), Layout(20, j_e), HeldAtFrw(), settings, engine=Engine.RUST)
            sch.evaluate_deviation(0.3, np.zeros(sch.layout.size))
            sch.evaluate(0.3, sch.frw(0.3))
    assert len(captured) == 8
    for name in ("stage_deviation", "stage_state"):  # both stage functions return a StageOutput
        annotation = returns.pop(name)
        assert annotation is not None
        for out in captured:
            assert stub_value_matches(out, ast.unparse(annotation), classes), name
    annotation = returns.pop("trapping")  # the finder, with a root inside and one outside
    assert annotation is not None
    U = np.array([0.0, -2.0, -2.0, -2.0, 0.5, -2.0, -2.0, 0.5, 1.0])
    out = pbh_engine.trapping(U, np.ones(9), 0)
    assert len(out.crossings) == 4
    assert stub_value_matches(out, ast.unparse(annotation), classes)
    annotation = returns.pop("checked_step")  # an attempt that arrives, and one refused at its second stage
    assert annotation is not None
    sch = Scheme(RAD, SinhStretch(6.0, scale=2.0), Layout(20), HeldAtFrw(), PRODUCTION_KERNELS, engine=Engine.RUST)
    dy = np.zeros(sch.layout.size)
    first = sch.evaluate_deviation(0.3, dy)
    k1 = sch.layout.pack(first.deviation_rate)
    a = [list(row) for row in RK4.floats[1]]
    b = list(RK4.floats[2])
    frames = [sch.frame(t) for t in (0.305, 0.305, 0.31)]
    bgs = [(f.bg.Gammabar2, f.bg.c_s, f.bg.hubble) for f in frames]
    rust_frames = [cast(pbh_engine.StageFrame, f.rust) for f in frames]
    settings = cast(RustStage, sch._rust)._settings  # pyright: ignore[reportPrivateUsage]
    for bad in (False, True):
        k = k1 + (1e300 if bad else 0.0)  # a stage input far outside the domain, refused
        out = pbh_engine.checked_step(settings, rust_frames, bgs, rust_frames[-1], bgs[-1], dy, 0.01, k, a, b)
        assert (out.failure is None) != bad
        assert stub_value_matches(out, ast.unparse(annotation), classes)
    annotation = returns.pop("blend_radii")
    assert annotation is not None
    blend = BlendMap(SinhStretch(6.0, scale=2.0), 0.5, (Zone(xi_on=0.4, tau_on=0.3, x_t=0.3, Delta_t=0.1),))
    radii = pbh_engine.blend_radii(*blend.static_part(20), *blend.ramps(0.6), blend.alpha)
    assert stub_value_matches(radii, ast.unparse(annotation), classes)
    case = violent_case("shock", 0)
    pair = case_pair(case, PRODUCTION_KERNELS)
    result = pair.numpy.evaluate(case.xi, pair.numpy.layout.pack(case.state))
    geo = pair.numpy.frame(case.xi).geo
    annotation = returns.pop("emptying_rates")
    assert annotation is not None
    rates = rust_engine.emptying_rates(result, case.state, geo, case.eos, case.layout)
    assert stub_value_matches(rates, ast.unparse(annotation), classes)
    annotation = returns.pop("near_zone")
    assert annotation is not None
    cells, faces = case.layout.cells, case.layout.faces
    d = result.derived
    fields = (geo.Xm[cells], geo.X[faces], d.ephi[cells], d.rho[cells], case.state.U[faces], d.Gammabar2[faces])
    assert stub_value_matches(rust_engine.near_zone(*fields, [1.0, 2.0]), ast.unparse(annotation), classes)
    assert not returns  # every function is covered
    assert {out.kernels is None for out in captured} == {True, False}

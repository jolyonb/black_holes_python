"""Tests of pbh.storage and what it changes: cells stored whole, the compensated clock, and the void's arithmetic.

A cell far below the background stores its content itself (Section 7.6, `storage.py`). The stage with such cells must
be the stage as printed (`whole_state.py`, which forms every flux whole and so is precise in a void) to round-off of
each void cell's own content, at depths from 1e-8 to 1e-30, on static and moving maps, excised and not, with both outer
closures; the deviation form given the same state is shown for contrast. Then the switch at a step boundary with its
hysteresis and its exact conversions, the clock that keeps advancing when the step is a few ulps of `xi`, the pressure
force that does not overflow near `rho ~ 1e-246`, and in the driver a restart from inside a void and one from a snapshot
taken with a carry, the clock-stall abort, a re-excision that keeps the cells stored whole, and files written before.
"""

from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
import pytest
from whole_state import whole_state_rate

from pbh import storage
from pbh.config import EvolutionConfig, GridConfig, MapFamily, OuterChoice, OuterConfig, OutputConfig, RunConfig
from pbh.driver import Run, RunPaths, run
from pbh.eos import RADIATION, Background, EquationOfState
from pbh.equations import calc_derivs, pressure_force
from pbh.excision import excise
from pbh.geometry import Geometry
from pbh.kernels import CENTRED_SCHEME, PRODUCTION_KERNELS, KernelSettings
from pbh.layout import Layout
from pbh.maps import SinhStretch
from pbh.monitors import MonitoredStep
from pbh.outer import PRODUCTION_STRENGTHS, HeldAtFrw, OuterClosure, OutgoingWave
from pbh.output import RunReader, RunWriter
from pbh.records import StateRecord, read_initial, write_initial
from pbh.state import State
from pbh.stencils import StencilWeights
from pbh.timestep import COMPENSATED_BELOW, Clock, StepChoice, StepLimit
from pbh.types import BoolArray, FloatArray

EOS = EquationOfState(RADIATION)
HELD = HeldAtFrw()
SAT = OutgoingWave(PRODUCTION_STRENGTHS)


def deviation_of(state: State, geo: Geometry, j_e: int) -> State:
    """The deviation from FRW as a state's own sums give it: rounded in the cells far below the background."""
    N = geo.N
    return State(E=state.E - geo.dV, U=state.U - geo.X[: N + 1], W=state.W, M_e=state.M_e - float(geo.X[j_e]) ** 3)


def stage(
    state: State,
    geo: Geometry,
    bg: Background,
    eos: EquationOfState,
    w: StencilWeights,
    outer: OuterClosure,
    settings: KernelSettings,
    whole: BoolArray | None,
):
    """The stage at `state`, its deviation formed from it, with the cells `whole` marks stored whole."""
    deviation = deviation_of(state, geo, w.layout.j_e)
    return calc_derivs(state, geo, bg, eos, w, outer, settings, deviation=deviation, whole=whole)


# --- the stage with cells stored whole is the stage as printed ---


def nonlinear(j_e: int, moving: bool) -> tuple[State, Geometry, Background, StencilWeights]:
    """A strongly nonlinear state on a sinh grid, densities from about 0.2 to 2, the map static or moving."""
    N, xi = 60, 0.8
    radii, _ = SinhStretch(6.0, scale=2.0).radii(xi, N)
    X_xi = 0.2 * radii * np.exp(-(radii**2)) if moving else np.zeros_like(radii)
    X_xi[N:] = 0.0
    geo = Geometry.of(radii, X_xi)
    X = geo.X[: N + 1]
    E = geo.dV * (1.0 + 1.2 * np.exp(-geo.sbar[:-1]) - 0.9 * np.exp(-((geo.Xm - 2.5) ** 2)))
    U = X * (1.0 - 0.3 * np.exp(-(X**2) / 2.0))
    state = State(E=E, U=U, W=0.03, M_e=1.4 * float(X[j_e]) ** 3)
    return state, geo, Background.at(EOS, xi), StencilWeights.of(geo, Layout(N, j_e))


@pytest.mark.parametrize("j_e", [0, 5])
@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("which", ["below_half", "every_other", "all_but_outer"])
def test_cells_stored_whole_give_the_printed_stage_and_the_deviation_forms(j_e: int, moving: bool, which: str):
    # Away from a void the two storages are the same arithmetic up to rounding: both agree with the printed stage.
    state, geo, bg, w = nonlinear(j_e, moving)
    N = w.layout.N
    rho = state.E / geo.dV
    whole = np.zeros(N, dtype=bool)
    if which == "below_half":
        whole[j_e:] = rho[j_e:] < 0.5
    elif which == "every_other":
        whole[j_e : N - storage.KEEP_DEVIATION : 2] = True
    else:
        whole[j_e : N - storage.KEEP_DEVIATION] = True
    assert whole.any()
    cells, faces = w.layout.cells, w.layout.faces
    for outer in (HELD, SAT):
        res = stage(state, geo, bg, EOS, w, outer, PRODUCTION_KERNELS, whole)
        old = stage(state, geo, bg, EOS, w, outer, PRODUCTION_KERNELS, None)
        ref = whole_state_rate(
            state, geo, bg, EOS, w, PRODUCTION_STRENGTHS if outer is SAT else None, PRODUCTION_KERNELS
        )
        assert np.max(np.abs(res.rate.E[cells] - ref.E[cells]) / geo.dV[cells]) < 1e-11
        assert np.max(np.abs(res.rate.E[cells] - old.rate.E[cells]) / geo.dV[cells]) < 1e-12
        scale = np.maximum(geo.X[faces], geo.X[1])
        assert np.nanmax(np.abs(res.rate.U[faces] - ref.U[faces]) / scale) < 1e-12
        assert np.nanmax(np.abs(res.rate.U[faces] - old.rate.U[faces]) / scale) < 1e-13
        assert res.rate.M_e == pytest.approx(ref.M_e, rel=1e-12, abs=1e-14)
        assert res.rate.W == old.rate.W
        # the stored rate: the whole rate where stored whole, the deviation rate elsewhere
        stored = res.stored_rate
        assert np.array_equal(stored.E[whole], res.rate.E[whole])
        near = np.flatnonzero(~whole[cells]) + j_e
        assert np.array_equal(stored.E[near], res.deviation_rate.E[near])
        assert np.array_equal(res.deviation_rate.E[whole], res.rate.E[whole] - geo.dV_xi[whole])
        assert stored.U is res.deviation_rate.U
        assert old.stored is None
        assert old.stored_rate is old.deviation_rate


def void_state(depth: float, j_e: int, moving: bool, eos: EquationOfState = EOS):
    """A band of cells emptying to `depth` of the background between dense walls, inside the Hubble radius.

    Outflow inside the band and infall outside it, as in the void inside a collapsing shell; on a moving map the faces
    in the band move. Returns the state, the geometry, the background, the weights and the cell densities.
    """
    N, xi = 120, 4.0
    radii, _ = SinhStretch(12.0, scale=3.0).radii(xi, N)
    X_xi = 0.3 * radii * np.exp(-(((radii - 2.0) / 1.5) ** 2)) if moving else np.zeros_like(radii)
    X_xi[N:] = 0.0
    geo = Geometry.of(radii, X_xi)
    X = geo.X[: N + 1]
    ln_rho = np.log(depth) * np.exp(-(((geo.Xm - 2.0) / 0.6) ** 4)) + 2.0 * np.exp(-(((geo.Xm - 3.2) / 0.3) ** 2))
    rho = np.exp(ln_rho)
    U = X * (1.0 + 0.4 * np.exp(-(((X - 2.0) / 0.7) ** 2)) - 0.5 * np.exp(-(((X - 3.4) / 0.4) ** 2)))
    state = State(E=rho * geo.dV, U=U, W=0.0, M_e=0.6 * float(X[j_e]) ** 3)
    return state, geo, Background.at(eos, xi), StencilWeights.of(geo, Layout(N, j_e)), rho


def below_quarter(rho: FloatArray, layout: Layout) -> BoolArray:
    """The cells the switch would store whole: below a quarter, retained, and not the outermost."""
    whole = np.zeros(layout.N, dtype=bool)
    whole[layout.j_e : layout.N - storage.KEEP_DEVIATION] = rho[layout.j_e : layout.N - storage.KEEP_DEVIATION] < 0.25
    return whole


@pytest.mark.parametrize("depth", [1e-8, 1e-14, 1e-16, 1e-20, 1e-30])
@pytest.mark.parametrize("j_e", [0, 4])
@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("outer", [HELD, SAT], ids=["held", "sat"])
def test_the_void_rates_are_precise_relative_to_the_void(depth: float, j_e: int, moving: bool, outer: OuterClosure):
    # The energy rate of each void cell, against the printed stage in the whole state, relative to the terms it is made
    # of (its two fluxes and its source: the best any evaluation of -(F_c+1 - F_c) + s E can do is their round-off).
    # The deviation form, given the same precise state, is noise there once rho is below about 1e-14, or NaN.
    state, geo, bg, w, rho = void_state(depth, j_e, moving)
    whole = below_quarter(rho, w.layout)
    void = np.flatnonzero(whole)
    res = stage(state, geo, bg, EOS, w, outer, PRODUCTION_KERNELS, whole)
    ref = whole_state_rate(state, geo, bg, EOS, w, PRODUCTION_STRENGTHS if outer is SAT else None, PRODUCTION_KERNELS)
    size = np.abs(ref.F[void]) + np.abs(ref.F[void + 1]) + np.abs(state.E[void])
    assert np.max(np.abs(res.rate.E[void] - ref.E[void]) / size) < 1e-14
    assert res.kernels is not None
    assert np.array_equal(res.F[void + 1], res.kernels.F[void + 1])
    assert np.max(np.abs(res.F[void + 1] - ref.F[void + 1]) / np.abs(ref.F[void + 1])) < 1e-13  # each flux, itself
    # every other row as the printed stage, to the round-off of the FRW sizes
    cells, faces = w.layout.cells, w.layout.faces
    assert np.max(np.abs(res.rate.E[cells] - ref.E[cells]) / geo.dV[cells]) < 1e-12
    scale = np.maximum(np.abs(ref.U[faces]), np.maximum(geo.X[faces], geo.X[1]))
    assert np.nanmax(np.abs(res.rate.U[faces] - ref.U[faces]) / scale) < 1e-11
    assert res.rate.M_e == pytest.approx(ref.M_e, rel=1e-12, abs=1e-14)
    with np.errstate(divide="ignore", invalid="ignore"):  # 1 + delta_rho rounds to zero below about 1e-16
        old = stage(state, geo, bg, EOS, w, outer, PRODUCTION_KERNELS, None)
    if depth <= 1e-14:
        assert not np.max(np.abs(old.rate.E[void] - ref.E[void]) / size) < 1e-2  # NaN or noise


@pytest.mark.parametrize("depth", [1e-8, 1e-20, 1e-30])
def test_the_centred_base_scheme_stores_cells_whole_as_printed(depth: float):
    state, geo, bg, w, rho = void_state(depth, 0, True)
    whole = below_quarter(rho, w.layout)
    void = np.flatnonzero(whole)
    res = stage(state, geo, bg, EOS, w, HELD, CENTRED_SCHEME, whole)
    ref = whole_state_rate(state, geo, bg, EOS, w, None, CENTRED_SCHEME)
    size = np.abs(ref.F[void]) + np.abs(ref.F[void + 1]) + np.abs(state.E[void])
    assert np.max(np.abs(res.rate.E[void] - ref.E[void]) / size) < 1e-14
    assert res.kernels is None
    assert res.F[0] == 0.0


@pytest.mark.parametrize("depth", [1e-8, 1e-14])
def test_cells_stored_whole_at_a_general_w(depth: float):
    # At w = 1/5 the deviation form's lapse is expm1(k log1p(delta_rho)); a cell stored whole takes rho^k - 1 instead.
    eos = EquationOfState(Fraction(1, 5))
    state, geo, bg, w, rho = void_state(depth, 0, False, eos)
    whole = below_quarter(rho, w.layout)
    void = np.flatnonzero(whole)
    res = stage(state, geo, bg, eos, w, HELD, PRODUCTION_KERNELS, whole)
    ref = whole_state_rate(state, geo, bg, eos, w, None, PRODUCTION_KERNELS)
    size = np.abs(ref.F[void]) + np.abs(ref.F[void + 1]) + np.abs(state.E[void])
    assert np.max(np.abs(res.rate.E[void] - ref.E[void]) / size) < 1e-13


def test_the_lapse_of_entries_far_below_the_background_is_formed_from_the_density():
    eos = EquationOfState(Fraction(1, 5))
    rho = np.array([1e-30, 0.9, 1e-300])
    delta_rho = rho - 1.0  # -1 exactly where rho is below the rounding of one
    whole = np.array([True, False, True])
    ephi, delta_ephi = eos.lapse_and_deviation(rho, delta_rho, whole)
    assert np.array_equal(ephi[whole], rho[whole] ** eos.lapse_exponent)
    assert np.array_equal(delta_ephi[whole], ephi[whole] - 1.0)
    near_ephi, near_delta = eos.lapse_and_deviation(rho[1:2], delta_rho[1:2])
    assert (ephi[1], delta_ephi[1]) == (near_ephi[0], near_delta[0])
    with np.errstate(divide="ignore"):
        assert np.isinf(eos.lapse_and_deviation(rho[:1], delta_rho[:1])[1][0])  # the deviation form's log1p(-1)


# --- the pressure force at extreme depth ---


def test_the_pressure_force_beside_a_cell_stored_whole_does_not_overflow():
    # <ephi> Gammabar^2 / <rho> overflows near rho ~ 1e-246; the ratio of force and density does not.
    lead = np.array([3.1e61, 2.0, 7.7e75])  # alpha / (1 + w) <ephi> Gammabar^2, <ephi> ~ rho^(-1/4)
    force = np.array([4.0e-248, 0.3, -2.5e-302])  # w D_s rho + Q, of the size of rho over a cell width
    rho_f = np.array([1.3e-247, 0.9, 6.0e-303])
    with np.errstate(over="ignore"):
        assert np.isinf(lead[0] / rho_f[0])
    beside = np.array([True, False, True])
    pressure = pressure_force(lead, force, rho_f, beside)
    exact = [-Fraction(a) * Fraction(f) / Fraction(r) for a, f, r in zip(lead, force, rho_f, strict=True)]
    assert np.all(np.isfinite(pressure))
    for p, e in zip(pressure, exact, strict=True):
        assert abs(Fraction(p) - e) <= abs(e) * Fraction(3, 2**53)
    assert pressure[1] == -(lead[1] / rho_f[1]) * force[1]  # away from a cell stored whole: the order as it was
    assert np.array_equal(pressure_force(lead[1:2], force[1:2], rho_f[1:2], None), pressure[1:2])


def test_a_void_at_1e_250_has_finite_rates_and_its_energy_rows_as_printed():
    state, geo, bg, w, rho = void_state(1e-250, 0, False)
    whole = below_quarter(rho, w.layout)
    void = np.flatnonzero(whole)
    res = stage(state, geo, bg, EOS, w, HELD, PRODUCTION_KERNELS, whole)
    cells, faces = w.layout.cells, w.layout.faces
    assert np.all(np.isfinite(res.rate.E[cells]))
    assert np.all(np.isfinite(res.rate.U[faces]))
    with np.errstate(over="ignore", invalid="ignore"):  # the printed velocity row overflows: only the energy is read
        ref = whole_state_rate(state, geo, bg, EOS, w, None, PRODUCTION_KERNELS)
    assert not np.all(np.isfinite(ref.U[faces]))
    size = np.abs(ref.F[void]) + np.abs(ref.F[void + 1]) + np.abs(state.E[void])
    assert np.max(np.abs(res.rate.E[void] - ref.E[void]) / size) < 1e-14


# --- the switch at a step boundary ---


def test_the_switch_has_a_gap_and_converts_exactly():
    layout = Layout(10, j_e=2)
    rng = np.random.default_rng(7)
    dV = rng.uniform(0.5, 3.0, 10)
    rho = np.array([np.nan, np.nan, 0.2, 0.3, 0.6, 0.4, 0.1, 2.0, 0.1, 0.1])  # cells 0, 1 excised; 8, 9 outermost
    whole = np.array([False, False, False, False, True, True, False, False, False, False])
    stored = np.where(whole, rho * dV, rho * dV - dV)
    E = np.full(10, np.nan)
    E[2:] = stored[2:]
    dy = layout.pack(State(E=E, U=np.full(11, 0.5), W=0.25, M_e=0.75))
    moved = storage.switched(dy, whole, rho, dV, layout)
    assert moved is not None
    dy_new, whole_new = moved
    assert whole_new is not None
    # below 1/4 to whole (2, 6); above 1/2 back (4); between the lines nothing moves (3 stays a deviation, 5 whole);
    # the excised cells and the outermost two never
    assert np.flatnonzero(whole_new).tolist() == [2, 5, 6]
    new = layout.unpack(dy_new)
    assert new.E[2] == dV[2] + E[2]
    assert new.E[6] == dV[6] + E[6]
    assert new.E[4] == E[4] - dV[4]
    assert [new.E[c] for c in (3, 5, 7, 8, 9)] == [E[c] for c in (3, 5, 7, 8, 9)]
    assert (new.W, new.M_e) == (0.25, 0.75)
    assert np.array_equal(new.U[2:], np.full(9, 0.5))
    assert storage.switched(dy_new, whole_new, rho, dV, layout) is None  # settled: a second look moves nothing
    # all back: no flags at all
    back = storage.switched(dy_new, whole_new, np.full(10, 0.9), dV, layout)
    assert back is not None
    assert back[1] is None


def test_the_conversions_across_the_lines_are_exact_by_sterbenz():
    rng = np.random.default_rng(11)
    dV = rng.uniform(1e-6, 1e3, 4000)
    # to whole below a half: Delta V + delta E with -Delta V <= delta E <= -Delta V / 2
    delta_E = -dV * rng.uniform(0.5, 1.0, dV.size)
    E = dV + delta_E
    assert all(Fraction(e) == Fraction(v) + Fraction(d) for e, v, d in zip(E, dV, delta_E, strict=True))
    # back to the deviation between a half and two: E - Delta V
    E = dV * rng.uniform(0.5, 2.0, dV.size)
    delta_E = E - dV
    assert all(Fraction(d) == Fraction(e) - Fraction(v) for d, e, v in zip(delta_E, E, dV, strict=True))


def test_the_storage_helpers():
    whole = np.array([False, True, False, False, True])
    assert np.flatnonzero(storage.faces_beside(whole)).tolist() == [1, 2, 4, 5]
    assert storage.any_whole(None) is None
    assert storage.any_whole(np.zeros(3, dtype=bool)) is None
    assert storage.any_whole(whole) is whole
    stored = State(E=np.array([0.5, 1e-30, -0.25, 0.0, 2e-20]), U=np.zeros(6), W=0.1, M_e=0.2)
    frw = State(E=np.ones(5), U=np.arange(6.0), W=0.0, M_e=1.0)
    state = storage.whole_of(stored, whole, frw)
    assert state.E.tolist() == [1.5, 1e-30, 0.75, 1.0, 2e-20]
    assert (state.W, state.M_e) == (0.1, 1.2)
    deviation = storage.deviation_of(stored, whole, frw.E)
    assert deviation.E.tolist() == [0.5, 1e-30 - 1.0, -0.25, 0.0, 2e-20 - 1.0]
    assert storage.whole_of(stored, None, frw).E.tolist() == frw.plus(stored).E.tolist()
    assert storage.deviation_of(stored, None, frw.E) is stored


# --- the compensated clock ---


def test_the_clock_is_the_plain_sum_while_no_step_is_short():
    clock = Clock(-3.0)
    for dxi in (0.13, 0.013, 1e-3, 2.5e-7, 0.1):
        nxt = clock.advanced(dxi)
        assert nxt == Clock(clock.xi + dxi)  # carry zero, xi the plain sum, to the bit
        clock = nxt
    assert Clock(0.0).advanced(1e-300) == Clock(1e-300)  # at xi = 0 nothing is short


def test_the_compensated_clock_keeps_the_sum_of_steps_of_a_few_ulps():
    xi = 6.2
    step = 4e-15  # about four and a half ulps of 6.2, the shortest step of the deep voids
    assert step < COMPENSATED_BELOW * xi
    clock, plain = Clock(xi), xi
    for _ in range(10000):
        clock, plain = clock.advanced(step), plain + step
    exact = Fraction(xi) + 10000 * Fraction(step)
    assert abs(Fraction(clock.xi) + Fraction(clock.carry) - exact) < Fraction(1e-29)
    assert abs(Fraction(clock.xi) - exact) <= Fraction(np.spacing(xi))  # xi itself to within an ulp of the sum
    assert abs(Fraction(plain) - exact) > 1000 * Fraction(np.spacing(xi))  # the plain sum drifts
    # below half an ulp the plain sum stops; the compensated clock goes on
    tiny = 1e-16
    assert xi + tiny == xi
    clock = Clock(xi)
    for _ in range(100):
        clock = clock.advanced(tiny)
    assert clock.xi > xi
    assert abs(Fraction(clock.xi) + Fraction(clock.carry) - Fraction(xi) - 100 * Fraction(tiny)) < Fraction(1e-30)
    # with a carry the clock stays compensated even for a step that is not short
    assert clock.carry != 0.0
    later = clock.advanced(0.01)
    assert Fraction(later.xi) + Fraction(later.carry) == Fraction(clock.xi) + Fraction(clock.carry + 0.01)


def test_a_step_below_the_rounding_of_the_carry_leaves_the_clock_where_it_was():
    clock = Clock(6.2).advanced(3e-16)
    assert clock.carry != 0.0
    assert clock.advanced(1e-40) == clock  # what the driver calls a stalled clock


# --- the driver: a restart from inside a void, the stall, and a re-excision ---


N = 40
VOID_CONFIG = RunConfig(
    grid=GridConfig(N=N, Rtilde_max=4.0, map=MapFamily.UNIFORM),
    outer=OuterConfig(closure=OuterChoice.HELD),
    output=OutputConfig(snapshot_spacing=0.01, snapshot_spacing_min=0.01),
    evolution=EvolutionConfig(xi_end=0.05),
)


def void_record(config: RunConfig, depth: float) -> StateRecord:
    """FRW at rest-relative with a band of cells 10..14 emptied to `depth`, held whole, outflow inside the band."""
    geo = config.scheme().frame(0.0).geo
    X = geo.X[: N + 1]
    delta_E = np.zeros(N)
    E_whole = np.full(N, np.nan)
    band = slice(10, 15)
    E_whole[band] = depth * geo.dV[band]
    delta_E[band] = E_whole[band] - geo.dV[band]
    delta_U = 0.05 * X * np.exp(-(((X - 1.2) / 0.3) ** 2))
    return StateRecord(delta_E, delta_U, 0.0, 0.0, X, 0.0, 0, {}, E_whole=E_whole)


def test_a_restart_from_inside_a_void_reproduces_the_run_bit_for_bit(tmp_path: Path):
    paths = RunPaths.of(tmp_path, "void")
    write_initial(paths.initial, void_record(VOID_CONFIG, 1e-30))
    initial = read_initial(paths.initial)
    assert initial.whole is not None
    assert np.flatnonzero(initial.whole).tolist() == [10, 11, 12, 13, 14]
    assert initial.state.E[12] == 1e-30 * VOID_CONFIG.scheme().frame(0.0).geo.dV[12]  # the content itself, read back
    result = run(VOID_CONFIG, initial, paths)
    assert result.status == "completed"
    reader = RunReader(paths.evolution)
    middle = reader.snapshot(2)
    assert middle.xi == 0.02
    assert middle.whole is not None
    assert middle.E_whole is not None
    assert np.min(middle.E_whole[middle.whole] / reader.geometry(0.02).dV[middle.whole]) < 1e-5  # still deep
    again = RunPaths.of(tmp_path, "again")
    write_initial(again.initial, middle)
    assert run(reader.config, read_initial(again.initial), again).status == "completed"  # with its saved start
    first, second = reader.snapshot(5), RunReader(again.evolution).snapshot(3)
    assert first.xi == second.xi == 0.05
    assert first.E_whole is not None
    assert second.E_whole is not None
    assert np.array_equal(first.E_whole, second.E_whole, equal_nan=True)
    assert np.array_equal(first.delta_E, second.delta_E)
    assert np.array_equal(first.delta_U, second.delta_U)
    steps, steps_again = reader.steps, RunReader(again.evolution).steps
    tail = np.asarray(steps["xi"]) > 0.02
    assert np.array_equal(np.asarray(steps["xi"])[tail], np.asarray(steps_again["xi"]))
    assert np.array_equal(np.asarray(steps["rho_0"])[tail], np.asarray(steps_again["rho_0"]))


def test_a_restart_from_a_snapshot_taken_with_a_carry_continues_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Unclipped steps of 0.006, one of 6e-19 among them, so that the snapshot at the first step past 0.01 is taken with
    # a carry; the run drops it there, as a restart from the snapshot starts without one, and the two then agree.
    config = VOID_CONFIG.model_copy(
        update={"output": OutputConfig(snapshot_spacing=0.01, snapshot_spacing_min=0.01, clip_to_snapshots=False)}
    )
    assert Clock(0.0).advanced(0.006).advanced(6e-19).advanced(0.006).carry != 0.0
    steps = iter([0.006, 6e-19])

    def choose(*_: object) -> StepChoice:
        return StepChoice(next(steps, 0.006), StepLimit.COURANT)

    monkeypatch.setattr("pbh.driver.step_size", choose)
    paths = RunPaths.of(tmp_path, "carry")
    assert run(config, void_record(config, 1e-3), paths).status == "completed"
    reader = RunReader(paths.evolution)
    taken = reader.snapshot(1)
    assert taken.xi == 0.012
    again = RunPaths.of(tmp_path, "again")
    write_initial(again.initial, taken)
    assert run(reader.config, read_initial(again.initial), again).status == "completed"  # with its saved start
    xi, xi_again = np.asarray(reader.steps["xi"]), np.asarray(RunReader(again.evolution).steps["xi"])
    assert np.array_equal(xi[xi > 0.012], xi_again)
    first, second = reader.snapshot(-1), RunReader(again.evolution).snapshot(-1)
    assert first.xi == second.xi == 0.05
    assert np.array_equal(first.delta_E, second.delta_E)
    assert np.array_equal(first.delta_U, second.delta_U)
    assert first.E_whole is not None
    assert second.E_whole is not None
    assert np.array_equal(first.E_whole, second.E_whole, equal_nan=True)


def test_a_file_written_before_cells_were_stored_whole_reads_as_none_stored_whole(tmp_path: Path):
    paths = RunPaths.of(tmp_path, "old")
    write_initial(paths.initial, void_record(VOID_CONFIG, 1e-3))
    assert run(VOID_CONFIG, read_initial(paths.initial), paths).status == "completed"
    with h5py.File(paths.initial, "r+") as f:
        del f["E_whole"]
    with h5py.File(paths.evolution, "r+") as f:
        del f["snapshots/E_whole"]
    old = read_initial(paths.initial)
    assert (old.E_whole, old.whole) == (None, None)
    assert np.array_equal(old.stored.E, old.delta_E)
    snapshot = RunReader(paths.evolution).snapshot(0)
    assert (snapshot.E_whole, snapshot.whole) == (None, None)


def test_a_clock_that_stalls_even_compensated_ends_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The first step lands on 0.01; then a step of 5e-19, below half an ulp, is kept in the carry, the next carries the
    # clock an ulp on, and one of 1e-40 is lost in the carry.
    steps = iter([0.01, 5e-19, 5e-19, 1e-40])

    def choose(*_: object) -> StepChoice:
        return StepChoice(next(steps), StepLimit.COURANT)

    monkeypatch.setattr("pbh.driver.step_size", choose)
    paths = RunPaths.of(tmp_path, "stall")
    result = run(VOID_CONFIG, void_record(VOID_CONFIG, 1e-3), paths)
    assert result.status == "aborted"
    reader = RunReader(paths.evolution)
    abort = [e for e in reader.events if e.kind == "abort"]
    assert len(abort) == 1
    assert abort[0].payload["field"] == "clock_stalled"
    end = reader.end
    assert end is not None
    assert "the clock stalled" in end.payload["reason"]
    xi = np.asarray(reader.steps["xi"])
    assert xi.tolist() == [0.01, 0.01, 0.01 + np.spacing(0.01)]  # the carry advanced the clock by an ulp
    assert reader.snapshots[-1].xi == xi[-1]  # the last good state


def test_a_re_excision_keeps_the_cells_stored_whole_outside_the_face(tmp_path: Path):
    config = VOID_CONFIG.model_copy(update={"outer": OuterConfig()})
    record = void_record(config, 1e-25)
    with RunWriter(tmp_path / "remap.evolution.h5", config, N, row_type=MonitoredStep) as writer:
        sch = config.scheme()
        r = Run(config, writer, sch, sch.layout.pack(record.stored), 0.0, None, (), 4.0, whole=record.whole)
        state = r.state()
        assert record.E_whole is not None
        assert state.E[12] == record.E_whole[12]
        excised, layout = excise(state, r.layout, 12)
        r.remap(layout, excised)
        assert r.whole is not None
        assert np.flatnonzero(r.whole).tolist() == [12, 13, 14]
        assert r.state().E[13] == state.E[13]  # the content itself carried over
        excised, layout = excise(r.state(), r.layout, 15)
        r.remap(layout, excised)
        assert r.whole is None  # the void inside the face: nothing stored whole remains

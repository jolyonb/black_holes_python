"""Tests of pbh.horizon: every marginally trapped sphere, the apparent horizon at fourth order, the margins."""

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from pbh.cli import main
from pbh.config import EvolutionConfig, ExcisionConfig, GridConfig, MapFamily, OutputConfig, RunConfig
from pbh.derived import Derived, derive
from pbh.driver import RunPaths, run
from pbh.eos import RADIATION, Background, EquationOfState
from pbh.geometry import Geometry
from pbh.horizon import HorizonReport, HorizonRow, NearZone, crossing, find_horizons, near_zone
from pbh.initial import cell_contents
from pbh.kernels import PRODUCTION_KERNELS
from pbh.layout import Layout
from pbh.maps import IdentityMap, Map, SinhStretch
from pbh.output import RunReader
from pbh.records import read_initial
from pbh.state import State, frw_state
from pbh.stencils import StencilWeights
from pbh.types import FloatArray

THETA = PRODUCTION_KERNELS.theta  # the theta-limiter fraction, which fixes the outer face density

RAD = EquationOfState(RADIATION)
E_XI = 25.0
XI = math.log(E_XI)  # e^xi = 25: on a box of radius 4, well inside e^(xi/2), the exterior can be untrapped
type Profile = Callable[[FloatArray], FloatArray]

# On an admissible slice a face is trapped where U < 0 and X^2 mt > e^xi, that is M(X) > e^xi X with M = 3 int x^2 rho:
# at a boundary U = -Gammabar, so Gammabar^2 = e^xi + U^2 - X^2 mt gives X^2 mt = e^xi whatever U is. Inside, the
# infall must be strong enough for Gammabar^2 > 0. A trapped shell therefore needs a dense shell followed by an
# underdense one, so that M(X) crosses the line e^xi X twice; the density is prescribed and the mass follows.


def density_state(m: Map, N: int, rho: Profile, v: float) -> tuple[State, Geometry, Background]:
    """A state with the density profile `rho` (contents by quadrature, exact for smooth profiles) and `U = v X`."""
    geo = Geometry.of(*m.radii(XI, N))
    X = geo.X[: N + 1]
    E = cell_contents(lambda Xq: rho(Xq) - 1.0, geo)
    U = v * X
    return State(E=E, U=U, W=0.0), geo, Background.at(RAD, XI)


def exact_roots(rho: Profile, brackets: list[tuple[float, float]]) -> list[float]:
    """The marginally trapped spheres of the density profile: the roots of `M(X) = e^xi X`."""

    def f(X: float) -> float:
        return 3.0 * quad(lambda x: x * x * float(rho(np.array([x]))[0]), 0.0, X, limit=200)[0] - E_XI * X

    return [float(brentq(f, lo, hi)) for lo, hi in brackets]


def report_for(m: Map, N: int, rho: Profile, v: float) -> tuple[HorizonReport, Geometry]:
    state, geo, bg = density_state(m, N, rho, v)
    layout = Layout(N)
    d = derive(state, geo, bg, RAD, StencilWeights.of(geo, layout), THETA)
    return find_horizons(state, d, geo, bg, RAD, m, layout, XI), geo


def one_shell(X: FloatArray) -> FloatArray:
    """A dense shell at 2.5 and an underdense one at 3.6: M crosses 25 X upward near 3.03 and downward near 3.47."""
    return 1.0 + 3.5 * np.exp(-((X - 2.5) ** 2) / 0.2) - 0.7 * np.exp(-((X - 3.6) ** 2) / 0.3)


# --- FRW ---


def test_frw_has_no_trapped_face_no_horizon_and_a_margin_of_one_at_the_origin():
    m = IdentityMap(4.0)
    geo = Geometry.of(*m.radii(0.5, 40))
    bg = Background.at(RAD, 0.5)
    state = frw_state(geo)
    layout = Layout(40)
    d = derive(state, geo, bg, RAD, StencilWeights.of(geo, layout), THETA)
    report = find_horizons(state, d, geo, bg, RAD, m, layout, 0.5)
    assert report.trapped_faces == 0
    assert report.horizons == ()
    assert report.apparent is None
    assert np.isnan(report.M_AH)
    assert np.isnan(report.residual)
    assert (report.margin, report.margin_face) == (1.0, 0)  # face 0: U = 0
    assert report.core_margin == 1.0  # FRW expands everywhere: the core is the origin alone
    assert not report.outer_face_trapped
    assert np.all(report.h[1:] > 0.0)
    near = near_zone(state, d, geo, report, RAD, layout, 0.5)
    row = HorizonRow.of(3, 0.5, report, None, near)
    assert (row.trapped_faces, row.horizons, row.j_star) == (0, 0, -1)
    assert np.isnan(row.lapse_AH)  # no horizon, no near zone
    assert row.min_lapse == 1.0  # but the lapse is recorded from the start: exactly one on FRW
    assert np.isnan(row.x_AH)
    assert np.isnan(row.zone_ratio)


# --- one trapped shell, from a density profile: the physics, and the accuracy of the interpolation ---


@pytest.mark.parametrize("m", [IdentityMap(4.0), SinhStretch(4.0, scale=2.0)])
def test_a_trapped_shell_has_an_inner_and_an_outer_boundary_found_at_fourth_order(m: Map):
    inner_exact, outer_exact = exact_roots(one_shell, [(2.5, 3.2), (3.2, 3.9)])
    errors: list[float] = []
    for N in (100, 200, 400):
        report, geo = report_for(m, N, one_shell, v=-1.5)
        assert len(report.horizons) == 2
        inner, outer = report.horizons
        assert not inner.outer
        assert outer.outer
        assert report.apparent == outer
        assert report.trapped_faces > 0
        assert not report.outer_face_trapped
        # the cubic root: fourth order in the local cell width (the linear root was second order, 1.0 dX^2)
        assert inner.X == pytest.approx(inner_exact, abs=2.0 * geo.dX[inner.j] ** 4)
        assert outer.X == pytest.approx(outer_exact, abs=2.0 * geo.dX[outer.j] ** 4)
        # the label and the radius agree through the analytic map, and the mass is the printed formula
        assert m.radius_at(XI, np.array([outer.x]))[0] == outer.X
        assert report.M_AH == pytest.approx(0.5 * math.exp(float(RAD.alpha) * XI) * outer.X)
        assert abs(report.residual) < 5e-3  # 2m/R - 1 at the interpolated horizon
        errors.append(abs(outer.X - outer_exact))
    assert errors[2] < errors[0] / 64.0  # measured 235 to 270 over two doublings; fourth order would be 256
    assert abs(report.residual) < 5e-4
    row = HorizonRow.of(7, XI, report, 0.5, NearZone(*[math.nan] * 8))
    assert (row.j_star, row.x_AH, row.X_AH, row.M_AH) == (outer.j, outer.x, outer.X, report.M_AH)
    assert row.zone_ratio == pytest.approx(outer.x / 0.5)


# --- the root inside a cell ---


def test_the_crossing_is_the_root_of_the_cubic_through_the_four_faces_around_it():
    # h a cubic in the label: the cubic through faces j-1..j+2 is h itself, so the root is exact to round-off.
    for root in (0.0, 1e-9, 0.37, 0.8, 1.0 - 1e-9):
        k = np.arange(10.0)
        h = (k - 4.0 - root) * (1.0 + 0.3 * (k - 4.0) + 0.05 * (k - 4.0) ** 2)
        assert crossing(h, 4, 0, 9) == pytest.approx(root, abs=1e-14)
        assert crossing(-h, 4, 0, 9) == pytest.approx(root, abs=1e-14)  # the other sign


def test_the_crossing_is_linear_where_the_four_faces_are_not_all_retained():
    h = np.array([0.5, -1.0, 2.0, 4.0, 9.0])
    assert crossing(h, 1, 1, 4) == pytest.approx(1.0 / 3.0, rel=1e-15)  # face 0 below the first retained face
    h = np.array([9.0, 4.0, 2.0, -1.0, 0.5])
    assert crossing(h, 3, 0, 4) == pytest.approx(2.0 / 3.0, rel=1e-15)  # the other end: face j + 2 beyond face N
    h = np.array([2.0, 1.0, -1.0, 1.0])
    assert crossing(h, 2, 0, 3) == pytest.approx(0.5, rel=1e-15)


def test_the_crossing_stays_in_the_cell_when_the_cubic_is_wild():
    # outer faces far off: the cubic swings hard, and Newton from the linear root must still end inside [0, 1] on a root
    h = np.array([-50.0, 1.0, -1.0, 60.0])
    t = crossing(h, 1, 0, 3)
    a1, a2, a3 = (
        -h[0] / 3 - h[1] / 2 + h[2] - h[3] / 6,
        h[0] / 2 - h[1] + h[2] / 2,
        -h[0] / 6 + h[1] / 2 - h[2] / 2 + h[3] / 6,
    )
    assert 0.0 <= t <= 1.0
    assert abs(h[1] + t * (a1 + t * (a2 + t * a3))) < 1e-12


# --- several trapped regions: the finder's logic on a prescribed trapping function ---


def test_every_sign_change_is_reported_and_the_outermost_outer_boundary_is_the_apparent_horizon():
    # The finder reads U, Gammabar^2, rho and M; with Gammabar = 1 and U = h - 1 it sees exactly the function h.
    N = 400
    m = IdentityMap(4.0)
    geo = Geometry.of(*m.radii(XI, N))
    X = geo.X[: N + 1]
    h = np.cos(np.pi * X)  # roots at 0.5, 1.5, 2.5, 3.5: trapped on (0.5, 1.5) and (2.5, 3.5)
    state = State(E=geo.dV.copy(), U=h - 1.0, W=0.0)
    d = Derived(
        rho=np.ones(N),
        ephi=np.ones(N),
        delta_ephi=np.zeros(N),
        M=X**3,
        delta_M=np.zeros(N + 1),
        mt=np.ones(N + 1),
        delta_rho=np.zeros(N),
        delta_U=np.zeros(N + 1),
        delta_m=np.zeros(N + 1),
        Gammabar2=np.ones(N + 1),
        rho_f=np.ones(N + 1),
        ephi_f=np.ones(N + 1),
        delta_rho_f=np.zeros(N + 1),
        delta_ephi_f=np.zeros(N + 1),
    )
    report = find_horizons(state, d, geo, Background.at(RAD, XI), RAD, m, Layout(N), XI)
    assert [round(s.X, 6) for s in report.horizons] == pytest.approx([0.5, 1.5, 2.5, 3.5], abs=2e-4)
    assert [s.outer for s in report.horizons] == [False, True, False, True]
    assert report.apparent == report.horizons[-1]
    assert report.trapped_faces == int(np.sum(h[1:] < 0.0))
    assert not report.outer_face_trapped
    assert report.margin == pytest.approx(np.min(h[1:]))  # 1 + U / Gammabar = h here
    assert report.M_AH == pytest.approx(0.5 * math.exp(0.5 * XI) * 3.5, abs=1e-3)


def test_a_trapped_outer_face_is_reported():
    report, _ = report_for(IdentityMap(4.0), 100, lambda X: 4.0 * np.ones_like(X), v=-3.0)
    assert report.outer_face_trapped  # M = 4 X^3 exceeds 25 X beyond X = 2.5, out through the outer face
    assert len(report.horizons) == 1
    assert not report.horizons[0].outer  # only the inner boundary exists on the grid ...
    assert report.apparent is None  # ... so there is no apparent horizon on it


def test_the_core_margin_is_over_the_central_infall_region():
    def core(X: FloatArray) -> FloatArray:
        return 1.0 + 3.0 * np.exp(-(X**2) / 0.25)  # central density 4

    N = 200
    m = IdentityMap(4.0)
    geo = Geometry.of(*m.radii(XI, N))
    X = geo.X[: N + 1]
    E = cell_contents(lambda Xq: core(Xq) - 1.0, geo)
    bg = Background.at(RAD, XI)
    layout = Layout(N)

    def report_with(U: FloatArray) -> tuple[HorizonReport, FloatArray]:
        state = State(E=E, U=U, W=0.0)
        d = derive(state, geo, bg, RAD, StencilWeights.of(geo, layout), THETA)
        return find_horizons(state, d, geo, bg, RAD, m, layout, XI), 1.0 + state.U / np.sqrt(d.Gammabar2)

    # infall at the centre, expansion beyond X ~ 0.9, and a faster infalling shell at X = 2.5
    U = X * (0.2 - 0.5 * np.exp(-(X**2) / 0.5) - 0.6 * np.exp(-((X - 2.5) ** 2) / 0.1))
    report, margin = report_with(U)
    end = int(np.flatnonzero(U > 0.0)[0])  # the first expanding face: the core is the faces before it
    assert 0.6 < X[end] < 1.1
    assert report.margin == pytest.approx(np.min(margin))
    assert 2.2 < X[report.margin_face] < 2.8  # the global minimum sits in the outer shell ...
    assert report.core_margin == pytest.approx(np.min(margin[:end]))  # ... the core's in the central infall
    assert report.core_margin_face < end
    assert report.margin < report.core_margin < 1.0
    # an expanding centre: the core is the first face alone, the origin, whose margin is one
    report, _ = report_with(X * (0.2 - 0.6 * np.exp(-((X - 2.5) ** 2) / 0.1)))
    assert (report.core_margin, report.core_margin_face) == (1.0, 0)
    assert report.margin < 1.0


# --- a real collapse: formation is an event, the horizon table fills, the margin history is recorded ---


@pytest.mark.slow
def test_a_collapse_records_its_formation_and_its_horizon_history(tmp_path: Path):
    config = RunConfig(
        grid=GridConfig(N=200, Rtilde_max=12.0, map=MapFamily.SINH, scale=3.0),
        output=OutputConfig(snapshot_spacing=0.5),
        excision=ExcisionConfig(enabled=False),  # the finder alone: the collapse runs on until the interior breaks
        evolution=EvolutionConfig(xi_end=6.0),
    )
    path = tmp_path / "bh.yaml"
    path.write_text(
        "grid: {N: 200, Rtilde_max: 12.0, scale: 3.0}\noutput: {snapshot_spacing: 0.5}\nevolution: {xi_end: 6.0}\n"
    )
    A = 0.515 * math.e / 8.0  # peak compaction 0.515 at ell = 2: a few per cent above threshold
    args = [
        "initial",
        "gaussian",
        "bh",
        "--config",
        str(path),
        "--A",
        f"{A:.12g}",
        "--ell",
        "2.0",
        "--dir",
        str(tmp_path),
    ]
    assert main(args) == 0
    paths = RunPaths.of(tmp_path, "bh")
    result = run(config, read_initial(paths.initial), paths)
    assert result.status == "aborted"  # unexcised, the interior breaks the areal coordinate after formation
    reader = RunReader(paths.evolution)
    kinds = [e.kind for e in reader.events]
    # The step that breaks the chart is tried at two halvings more, three `rejection`s, before the run aborts.
    assert kinds == ["formation", "rejection", "rejection", "rejection", "abort", "end"]
    end = reader.end
    assert end is not None
    assert "a trapped region the excision has not caught" in end.payload["reason"]
    formation = reader.events[0]
    assert 4.3 < formation.xi < 5.0
    assert formation.payload["M_AH"] > 0.0
    assert formation.payload["j_star"] >= 1
    table = reader.horizon
    xi = np.asarray(table["xi"], dtype=np.float64)
    assert len(xi) == result.steps + 1  # the examination of the initial state, then one per step
    trapped = np.asarray(table["trapped_faces"], dtype=np.int64)
    first = int(np.flatnonzero(trapped > 0)[0])
    assert xi[first] == formation.xi
    assert np.all(trapped[:first] == 0)
    M_AH = np.asarray(table["M_AH"], dtype=np.float64)
    assert np.all(np.isnan(M_AH[:first]))
    assert np.all(np.diff(M_AH[first:]) >= 0.0)  # the apparent-horizon mass is monotone, as the paper says
    margin = np.asarray(table["margin"], dtype=np.float64)
    core_margin = np.asarray(table["core_margin"], dtype=np.float64)
    assert core_margin[0] > 0.9  # far from trapped at the start ...
    assert core_margin[first - 1] < 0.6  # ... falling toward formation
    assert margin[first - 1] > 0.0 > margin[first]  # the global margin crosses zero at formation, and the core's
    assert core_margin[first] == margin[first]  # with it: the trapped shell lies in the central infall region
    # the snapshots carry the formation time from then on, and the schedule switched to its post-formation branch
    infos = reader.snapshots
    after = [s for s in infos if s.xi > formation.xi]
    assert after
    assert reader.snapshot(after[0].index).xi_form == formation.xi


def test_a_run_whose_outer_face_becomes_trapped_aborts_as_a_result(tmp_path: Path):
    # A box entirely inside a trapped region: dense, strongly infalling. The finder sees face N trapped after
    # the first step and the driver ends the run with the abort recorded.
    config = RunConfig(
        grid=GridConfig(N=60, Rtilde_max=4.0, map=MapFamily.UNIFORM),
        output=OutputConfig(snapshot_spacing=0.5),
        evolution=EvolutionConfig(xi_end=XI + 0.05),
    )
    state, geo, _ = density_state(IdentityMap(4.0), 60, lambda X: 4.0 * np.ones_like(X), v=-3.0)
    from pbh.records import StateRecord, write_initial

    paths = RunPaths.of(tmp_path, "swallowed")
    write_initial(paths.initial, StateRecord.of(state, geo.X[:61], XI, 0, {"method": "test"}))
    result = run(config, read_initial(paths.initial), paths)
    assert result.status == "aborted"
    assert result.steps == 0  # seen at the examination before the first step
    reader = RunReader(paths.evolution)
    kinds = [e.kind for e in reader.events]
    assert kinds == ["abort", "end"]
    assert reader.events[0].payload["field"] == "outer_face_trapped"
    assert reader.events[0].payload["index"] == 60
    assert len(reader.snapshots) == 1  # the last good state, which is the initial one


def test_formation_is_recorded_on_the_first_step_with_a_trapped_face(tmp_path: Path):
    # A state that already holds a trapped shell inside an untrapped exterior: the finder sees it at the examination
    # before the first step, the driver writes the formation event, sets the formation time, and the table fills.
    from pbh.records import StateRecord, write_initial

    N = 100
    config = RunConfig(
        grid=GridConfig(N=N, Rtilde_max=4.0, map=MapFamily.UNIFORM),
        output=OutputConfig(snapshot_spacing=0.01),
        excision=ExcisionConfig(enabled=False),  # the finder's record alone: this hole is too big for the grid's zone
        evolution=EvolutionConfig(xi_end=XI + 0.02),
    )
    state, geo, _ = density_state(IdentityMap(4.0), N, one_shell, v=-1.5)
    paths = RunPaths.of(tmp_path, "shell")
    write_initial(paths.initial, StateRecord.of(state, geo.X[: N + 1], XI, 0, {"method": "test"}))
    result = run(config, read_initial(paths.initial), paths)
    reader = RunReader(paths.evolution)
    formation = reader.events[0]
    assert formation.kind == "formation"
    assert formation.step == 0  # the examination of the initial state
    assert formation.payload["j_star"] > 0
    assert 3.3 < formation.payload["X_AH"] < 3.6
    table = reader.horizon
    assert len(table["xi"]) == result.steps + 1  # the examination of the initial state, then one per step
    assert np.asarray(table["trapped_faces"])[0] > 0
    assert reader.snapshot(len(reader.snapshots) - 1).xi_form == formation.xi


def test_a_refused_switch_on_is_logged_and_the_run_goes_on_unexcised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The same trapped shell with excision on, every attempt refused for a stencil not yet trapped: the refusal is an
    # event, once, and the run steps on unexcised.
    import dataclasses

    import pbh.driver as driver_module
    from pbh.excision import SwitchAttempt
    from pbh.records import StateRecord, write_initial

    real = driver_module.attempt_switch_on

    def thin(*args: object, **kwargs: object) -> SwitchAttempt:
        attempt = real(*args, **kwargs)  # type: ignore[arg-type]
        return dataclasses.replace(attempt, three_trapped=False, transition_fits=True)

    monkeypatch.setattr(driver_module, "attempt_switch_on", thin)
    N = 100
    config = RunConfig(
        grid=GridConfig(N=N, Rtilde_max=4.0, map=MapFamily.UNIFORM),
        output=OutputConfig(snapshot_spacing=0.01),
        evolution=EvolutionConfig(xi_end=XI + 0.02),
    )
    state, geo, _ = density_state(IdentityMap(4.0), N, one_shell, v=-1.5)
    paths = RunPaths.of(tmp_path, "thin")
    write_initial(paths.initial, StateRecord.of(state, geo.X[: N + 1], XI, 0, {"method": "test"}))
    result = run(config, read_initial(paths.initial), paths)
    reader = RunReader(paths.evolution)
    attempts = [e for e in reader.events if e.kind == "switch_attempt"]
    assert len(attempts) == 1
    assert attempts[0].payload["failed"] == ["three_trapped"]
    assert result.steps >= 1

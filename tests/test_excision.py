"""Tests of pbh.excision on hand-built states and on a real collapse: the switch-on tests, excising, the
assertions, re-excision, and the packing of an excised deviation."""

import dataclasses
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pbh.cli import main
from pbh.config import ExcisionConfig, RunConfig, load
from pbh.derived import derive
from pbh.driver import READOUT_CHECK, RunPaths, run
from pbh.eos import RADIATION, Background, EquationOfState
from pbh.excision import (
    FOLD_DETECTION,
    OUTER_STATIC_LABEL,
    ExcisionError,
    SwitchAttempt,
    attempt_switch_on,
    check_face,
    check_fold,
    excise,
    fold_monitor,
    fold_numbers,
    label_of,
    outflow_margin,
    packed_deviation,
    place_transition,
    re_excision_face,
    taub_velocity,
    zone_needs_extension,
)
from pbh.geometry import Geometry
from pbh.horizon import (
    NO_FOLD,
    UNEXCISED,
    FaceValues,
    FoldValues,
    Horizon,
    HorizonReport,
    HorizonRow,
    find_horizons,
    near_zone,
)
from pbh.kernels import PRODUCTION_KERNELS
from pbh.layout import Layout
from pbh.maps import BlendMap, IdentityMap, SinhStretch, Zone
from pbh.michel import michel_flow
from pbh.output import RunReader
from pbh.records import read_initial
from pbh.state import frw_state
from pbh.stencils import StencilWeights
from pbh.summary import as_json, describe, summarise
from pbh.timestep import Scheme

THETA = PRODUCTION_KERNELS.theta  # the theta-limiter fraction, which fixes the outer face density

RAD = EquationOfState(RADIATION)
EXCISION = ExcisionConfig()


# --- a collapse to work on: the N = 200 Gaussian of the finder test, dense snapshots after formation ---


@pytest.fixture(scope="module")
def collapse(tmp_path_factory: pytest.TempPathFactory) -> tuple[RunReader, RunConfig]:
    directory = tmp_path_factory.mktemp("collapse")
    path = directory / "bh.yaml"
    path.write_text(
        "grid: {N: 200, Rtilde_max: 12.0, scale: 3.0}\n"
        "output: {snapshot_spacing: 0.5, snapshot_spacing_after: 0.01}\n"
        "excision: {enabled: false}\n"
        "evolution: {xi_end: 6.0}\n"
    )
    A = 0.515 * math.e / 8.0
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
        str(directory),
    ]
    assert main(args) == 0
    config = load(path)
    paths = RunPaths.of(directory, "bh")
    run(config, read_initial(paths.initial), paths)
    return RunReader(paths.evolution), config


def slice_of(reader: RunReader, config: RunConfig, index: int):
    """A snapshot's state with everything the excision functions read, on the unexcised layout."""
    record = reader.snapshot(index)
    sch = config.scheme()
    frame = sch.frame(record.xi)
    layout = Layout(config.grid.N)
    state = record.state
    d = derive(state, frame.geo, frame.bg, RAD, StencilWeights.of(frame.geo, layout), THETA)
    report = find_horizons(state, d, frame.geo, frame.bg, RAD, sch.map, layout, record.xi)
    return record, state, d, frame.geo, frame.bg, report, layout


def scheme_on(config: RunConfig, layout: Layout) -> Scheme:
    """The configuration's scheme on an excised layout."""
    return Scheme(RAD, config.grid.build(), layout, config.outer.build(), config.shocks.build())


def formed_snapshots(reader: RunReader, config: RunConfig) -> list[int]:
    """The indices of the snapshots with a trapped face."""
    return [s.index for s in reader.snapshots if slice_of(reader, config, s.index)[5].trapped_faces > 0]


# --- the outflow margin ---


def test_frw_faces_are_not_outflow_faces_and_trapped_faces_are():
    geo = Geometry.of(*IdentityMap(4.0).radii(0.5, 40))
    bg = Background.at(RAD, 0.5)
    state = frw_state(geo)
    layout = Layout(40)
    d = derive(state, geo, bg, RAD, StencilWeights.of(geo, layout), THETA)
    # mu = alpha [X - <ephi> (U + Gammabar)] with U = X and ephi = 1 on FRW: -alpha Gammabar < 0, light escapes
    for j in (1, 10, 39):
        assert outflow_margin(j, state, d, geo, RAD, layout) == pytest.approx(-0.5 * math.sqrt(bg.Gammabar2))


def test_the_switch_on_is_refused_while_the_shell_is_thin_and_thrown_once_it_has_thickened(
    collapse: tuple[RunReader, RunConfig],
):
    reader, config = collapse
    formed = formed_snapshots(reader, config)
    assert formed
    record, state, d, geo, _, report, layout = slice_of(reader, config, formed[-1])
    grid = config.grid.build()
    attempt = attempt_switch_on(report, state, d, geo, RAD, layout, EXCISION, (), grid, record.xi)
    assert isinstance(attempt, SwitchAttempt)
    assert attempt.passed
    assert attempt.failed == []
    assert attempt.mu > 0.0
    assert all(h < 0.0 for h in attempt.h)
    assert 1 <= attempt.j_e < attempt.j_star
    assert attempt.j_e == math.ceil(layout.N * attempt.x_e)
    assert attempt.x_e == pytest.approx(EXCISION.eta * math.exp(-0.5 * EXCISION.tau_on) * attempt.x_AH)
    assert attempt.x_t + attempt.Delta_t < OUTER_STATIC_LABEL
    assert attempt.zone(xi_on=1.0, tau_on=0.3) == Zone(1.0, 0.3, attempt.x_t, attempt.Delta_t)
    # the same slice with the trapped region thinned to a shell just inside the horizon, as it is at first
    # detection: the candidate face's stencil is not trapped, and that is the test that refuses the switch
    h = report.h.copy()
    h[: attempt.j_e + 2] = np.abs(h[: attempt.j_e + 2])
    thin = dataclasses.replace(report, h=h, trapped_faces=int(np.sum(h < 0.0)))
    refused = attempt_switch_on(thin, state, d, geo, RAD, layout, EXCISION, (), grid, record.xi)
    assert not refused.passed
    assert refused.failed == ["three_trapped"]
    assert refused.inside_horizon
    assert refused.margin_positive


def test_a_repeated_switch_on_must_move_the_face_outward_and_must_not_overlap_the_zone(
    collapse: tuple[RunReader, RunConfig],
):
    reader, config = collapse
    index = formed_snapshots(reader, config)[-1]
    record, state, d, geo, _, report, layout = slice_of(reader, config, index)
    grid = config.grid.build()
    attempt = attempt_switch_on(report, state, d, geo, RAD, layout, EXCISION, (), grid, record.xi)
    assert attempt.passed
    # the same slice seen from a run already excised beyond the candidate face: the face stays where it is
    excised_state, excised_layout = excise(state, layout, attempt.j_e + 2)
    again = attempt_switch_on(
        report, excised_state, d, geo, RAD, excised_layout, EXCISION, (), config.grid.build(), record.xi
    )
    assert again.j_e == attempt.j_e + 2
    assert again.inside_horizon
    # with an existing zone the new transition starts beyond its end, so the two never overlap ...
    earlier = Zone(xi_on=0.0, tau_on=0.3, x_t=attempt.x_t - attempt.Delta_t, Delta_t=0.5 * attempt.Delta_t)
    extension = attempt_switch_on(
        report, state, d, geo, RAD, layout, EXCISION, (earlier,), config.grid.build(), record.xi
    )
    assert extension.no_overlap
    assert extension.x_t - extension.Delta_t == pytest.approx(earlier.outer_edge)
    # ... unless that puts it past the static outer part, which the fourth test refuses
    far = Zone(xi_on=0.0, tau_on=0.3, x_t=0.6, Delta_t=0.1)
    assert not attempt_switch_on(
        report, state, d, geo, RAD, layout, EXCISION, (far,), config.grid.build(), record.xi
    ).transition_fits


def test_an_extension_starts_exactly_where_the_last_zone_ends_whatever_the_rounding():
    # The old zone ends at a = 0.11, and for a width b = 0.261372 the sum (a + b) - b rounds below a: an extension
    # placed by the plain sum would overlap the old zone by an ulp, which the map refuses.
    earlier = Zone(xi_on=0.0, tau_on=0.3, x_t=0.08, Delta_t=0.03)
    a, b = earlier.outer_edge, 0.261372
    assert (a + b) - b < a
    x_t = place_transition(0.0, b, (earlier,))
    assert x_t - b >= earlier.outer_edge
    assert x_t - b - earlier.outer_edge < 1e-16
    BlendMap(IdentityMap(1.0), 0.5, (earlier, Zone(xi_on=1.0, tau_on=0.3, x_t=x_t, Delta_t=b)))  # the map accepts it
    assert place_transition(0.3, b, ()) == 0.3  # the first zone goes where it is sized
    assert place_transition(0.9, b, (earlier,)) == 0.9  # an extension already clear of the old zone stays put


# --- excising ---


def test_excising_keeps_the_cumulative_mass_at_every_retained_face(collapse: tuple[RunReader, RunConfig]):
    reader, config = collapse
    index = formed_snapshots(reader, config)[-1]
    _, state, d, geo, bg, _, layout = slice_of(reader, config, index)
    j_e = 12
    excised, excised_layout = excise(state, layout, j_e)
    assert excised_layout.j_e == j_e
    assert np.all(np.isnan(excised.E[:j_e]))
    assert np.all(np.isnan(excised.U[:j_e]))
    assert excised.M_e == pytest.approx(3.0 * np.sum(state.E[:j_e]))
    d_excised = derive(excised, geo, bg, RAD, StencilWeights.of(geo, excised_layout), THETA)
    assert d_excised.M[j_e:] == pytest.approx(d.M[j_e:], rel=1e-14)
    assert d_excised.mt[j_e:] == pytest.approx(d.mt[j_e:], rel=1e-14)
    # a second move, further out, adds the dropped cells to the mass inside
    further, further_layout = excise(excised, excised_layout, j_e + 5)
    assert further.M_e == pytest.approx(excised.M_e + 3.0 * np.sum(state.E[j_e : j_e + 5]))
    assert further_layout.j_e == j_e + 5
    with pytest.raises(ValueError, match="only move outward"):
        excise(excised, excised_layout, j_e)
    with pytest.raises(ValueError, match="only move outward"):
        excise(excised, excised_layout, layout.N)
    # the packed deviation of the excised state unpacks to it, with the mass against its FRW value X_e^3
    dy = packed_deviation(excised, geo, excised_layout, geo.dV)
    back = excised_layout.unpack(dy)
    assert back.E[j_e:] == pytest.approx(excised.E[j_e:] - geo.dV[j_e:], rel=1e-12)
    assert back.M_e == pytest.approx(excised.M_e - geo.X[j_e] ** 3)
    assert back.W == excised.W


# --- the assertions of every excised step ---


def test_the_face_assertions_pass_inside_the_trapped_region_and_fail_outside_it(collapse: tuple[RunReader, RunConfig]):
    reader, config = collapse
    index = formed_snapshots(reader, config)[-1]
    record, state, d, geo, bg, report, layout = slice_of(reader, config, index)
    attempt = attempt_switch_on(report, state, d, geo, RAD, layout, EXCISION, (), config.grid.build(), record.xi)
    assert attempt.passed
    excised, excised_layout = excise(state, layout, attempt.j_e)
    sch = Scheme(RAD, config.grid.build(), excised_layout, config.outer.build(), config.shocks.build())
    result = sch.evaluate(record.xi, excised_layout.pack(excised))
    d_e = result.derived
    report_e = find_horizons(excised, d_e, geo, bg, RAD, sch.map, excised_layout, record.xi)
    face = check_face(
        report_e, excised, d_e, result.speeds, float(result.F[attempt.j_e]), geo, bg, RAD, excised_layout, record.xi
    )
    assert isinstance(face, FaceValues)
    assert face.j_e == attempt.j_e
    assert face.mu > 0.0
    assert face.Lambda_plus == 0.0  # every characteristic points inward: the flux is fully upwind
    assert face.sound_margin > 0.0
    assert face.a_over_Theta < math.sqrt(1.0 / 3.0)  # below sqrt w, where the closure is certified
    assert all(h < 0.0 for h in face.h)
    assert face.faces_to_horizon == attempt.j_star - attempt.j_e
    assert face.M_e == excised.M_e
    assert face.F_e == float(result.F[attempt.j_e])
    assert 0.0 < face.R_e_over_M_AH < 2.0  # inside the horizon, whose radius is 2 M_AH
    assert face.physical_margin > 0.0
    # no shock inside the horizon on the smooth collapse: the fold monitor measures its largest jump, the infall across
    # the few cells to the horizon at N = 200, just short of the detection level and less than half way to a fold
    fold = face.fold
    assert fold == fold_monitor(excised, d_e, geo, RAD, excised_layout, report_e.M_AH, record.xi)
    assert fold.X_zone == pytest.approx(2.0 * report_e.M_AH * math.exp(-RAD.alpha_float * record.xi), rel=1e-15)
    assert fold.fronts == 0
    assert 1.0 <= fold.jump < FOLD_DETECTION
    assert attempt.j_e < fold.cell
    assert fold.X < fold.X_zone
    assert 1.0 <= fold.compression < 2.5
    assert fold.U1 < 0.0 < fold.Gammabar1
    assert 0.0 < fold.ratio < 0.5
    near = near_zone(excised, d_e, geo, report_e, RAD, excised_layout, record.xi)
    row = HorizonRow.of(5, record.xi, report_e, None, near, face)
    assert (row.j_e, row.mu, row.h_e2) == (face.j_e, face.mu, face.h[2])
    assert (row.fold_fronts, row.fold_jump, row.fold_cell, row.fold_ratio) == (0, fold.jump, fold.cell, fold.ratio)
    assert (row.fold_X_zone, row.fold_X, row.fold_U1) == (fold.X_zone, fold.X, fold.U1)
    assert (row.fold_Gammabar1, row.fold_compression, row.fold_v12) == (fold.Gammabar1, fold.compression, fold.v12)
    unexcised = HorizonRow.of(5, record.xi, report_e, None, near)
    assert unexcised.j_e == UNEXCISED.j_e == -1
    assert (unexcised.fold_fronts, unexcised.fold_cell) == (NO_FOLD.fronts, NO_FOLD.cell) == (0, -1)
    assert math.isnan(unexcised.fold_jump)
    # a face outside the trapped region fails the trapped-stencil assertion ...
    outside, outside_layout = excise(state, layout, attempt.j_star + 3)
    sch_out = Scheme(RAD, config.grid.build(), outside_layout, config.outer.build(), config.shocks.build())
    result_out = sch_out.evaluate(record.xi, outside_layout.pack(outside))
    report_out = find_horizons(outside, result_out.derived, geo, bg, RAD, sch.map, outside_layout, record.xi)
    with pytest.raises(ExcisionError, match="trapped"):
        check_face(
            report_out, outside, result_out.derived, result_out.speeds, 0.0, geo, bg, RAD, outside_layout, record.xi
        )


def test_an_excised_frw_face_fails_the_margin_assertion():
    geo = Geometry.of(*IdentityMap(4.0).radii(0.5, 40))
    bg = Background.at(RAD, 0.5)
    layout = Layout(40, j_e=5)
    state = frw_state(geo, 5)
    sch = Scheme(
        RAD,
        IdentityMap(4.0),
        layout,
        __import__("pbh.outer", fromlist=["HeldAtFrw"]).HeldAtFrw(),
        __import__("pbh.kernels", fromlist=["CENTRED_SCHEME"]).CENTRED_SCHEME,
    )
    result = sch.evaluate(0.5, layout.pack(state))
    report = find_horizons(state, result.derived, geo, bg, RAD, sch.map, layout, 0.5)
    with pytest.raises(ExcisionError, match="mu > 0"):
        check_face(report, state, result.derived, result.speeds, 0.0, geo, bg, RAD, layout, 0.5)


# --- the fold monitor (Section 8.3) ---


def test_the_taub_velocity_is_the_relation_the_paper_prints_at_every_w():
    # exactly: v_12^2 = s^2 / (1 + s^2), s^2 = w (r - 1)^2 / ((1 + w)^2 r), in Fractions
    for w, r in ((Fraction(1, 3), Fraction(4)), (Fraction(1, 3), Fraction(9, 5)), (Fraction(1, 10), Fraction(7, 2))):
        s2 = w * (r - 1) ** 2 / ((1 + w) ** 2 * r)
        assert taub_velocity(float(r), float(w)) ** 2 == pytest.approx(float(s2 / (1 + s2)), rel=1e-15)
    assert taub_velocity(4.0, 1.0 / 3.0) ** 2 == pytest.approx(27.0 / 91.0, rel=1e-15)
    assert taub_velocity(1.8, 1.0 / 3.0) == pytest.approx(0.25, rel=1e-15)
    assert taub_velocity(1.0, 1.0 / 3.0) == 0.0
    # against the shock-frame speeds of Section 8.6, v_1 = [w (r + w) / (1 + w r)]^(1/2), v_1 v_2 = w, composed
    for w in (0.1, 1.0 / 3.0, 0.5, 0.9):  # at w = 1 every front runs at light speed, v_1 = v_2 = 1
        for r in (1.2, 2.0, 5.5, 40.0):
            v1 = math.sqrt(w * (r + w) / (1.0 + w * r))
            v2 = w / v1
            assert taub_velocity(r, w) == pytest.approx((v1 - v2) / (1.0 - v1 * v2), rel=1e-13)


def compression_at(v12: float) -> float:
    """The radiation compression whose Taub velocity is `v12`, by bisection."""
    lo, hi = 1.0, 100.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if taub_velocity(mid, 1.0 / 3.0) < v12 else (lo, mid)
    return lo


def test_the_fold_thresholds_are_the_compressions_the_paper_quotes():
    # Trapped, on the Michel background: Gammabar/|U| = 1/|v| is 1 at 2M, 0.64 at 1.4M and 0.43 at M, which takes a
    # compression of about 5.5 at 1.4M and 2.85 at M (Section 6.2); anti-trapped, 1/(HR) at 1.5, 2 and 4 Hubble radii
    # takes 6.1, 3.5 and 1.8.
    flow = michel_flow(np.array([2.0, 1.4, 1.0]))
    threshold = flow.N / np.abs(flow.U)
    assert threshold == pytest.approx([1.0, 0.64, 0.43], abs=5e-3)
    assert round(compression_at(float(threshold[1])), 1) == 5.5
    assert round(compression_at(float(threshold[2])), 2) == 2.85
    assert [round(compression_at(1.0 / radius), 1) for radius in (1.5, 2.0, 4.0)] == [6.1, 3.5, 1.8]


FOLD_N, FOLD_JE = 40, 5
FOLD_GEO = Geometry.of(*IdentityMap(4.0).radii(0.0, FOLD_N))  # X_j = j / 10
FOLD_LAYOUT = Layout(FOLD_N, j_e=FOLD_JE)


def front(post: float, middle: float | None = None, at: int = 20, outward: bool = True) -> np.ndarray:
    """Cell densities with a front at face `at`: `post` inside, `1` outside, one intermediate cell if `middle`."""
    rho = np.full(FOLD_N, np.nan)
    rho[FOLD_JE:at], rho[at:] = (post, 1.0) if outward else (1.0, post)
    if middle is not None:
        rho[at] = middle
    return rho


def fold_of(rho: np.ndarray, Gammabar: float = 0.5, U: float = -1.0, X_zone: float = 3.0) -> FoldValues:
    """The fold monitor on the cell densities `rho`, with `Gammabar` and `U` uniform."""
    U_f = np.full(FOLD_N + 1, U)
    Gammabar2 = np.full(FOLD_N + 1, Gammabar**2)
    return fold_numbers(rho, U_f, Gammabar2, FOLD_GEO, FOLD_LAYOUT, X_zone, 1.0 / 3.0)


def test_a_front_is_detected_by_the_jump_across_a_cell_and_measured_across_its_cells():
    # a weak front, 1.4 across two cells, is below the detection level: it is measured, but no front is reported
    weak = fold_of(front(1.4, 1.2))
    assert (weak.fronts, weak.jump, weak.cell) == (0, pytest.approx(1.4), 20)
    assert weak.compression == pytest.approx(1.4)
    assert weak.ratio == weak.v12 / 0.5 < 1.0
    check_fold(weak)
    # a front of compression 4 captured over two cells: detected by the jump of 4 across the intermediate cell, one
    # front of three detected cells 19 .. 21, measured across them, the pre-shock state at the outer face of cell 22
    strong = fold_of(front(4.0, 2.5))
    assert (strong.fronts, strong.jump, strong.cell) == (1, 4.0, 20)
    assert strong.X == FOLD_GEO.Xm[20]
    assert strong.compression == 4.0
    assert strong.v12 == taub_velocity(4.0, 1.0 / 3.0)
    assert (strong.U1, strong.Gammabar1, strong.threshold) == (-1.0, 0.5, 0.5)
    assert strong.ratio == strong.v12 / 0.5
    # a sharp front: the jump is the compression
    sharp = fold_of(front(3.0))
    assert (sharp.fronts, sharp.jump, sharp.cell, sharp.compression) == (1, 3.0, 19, 3.0)
    # an inward-facing front (denser outside) cannot fold a trapped slice and is not sought
    assert fold_of(front(4.0, outward=False)).fronts == 0
    # a front beyond 2 M_AH is outside the zone; a zone with no cell beyond the excision face has nothing to measure
    assert fold_of(front(4.0, at=32)).fronts == 0
    empty = fold_of(front(4.0), X_zone=0.55)
    assert (empty.X_zone, empty.fronts, empty.cell) == (0.55, 0, -1)
    assert all(math.isnan(v) for v in (empty.jump, empty.X, empty.compression, empty.threshold, empty.ratio))
    check_fold(empty)
    # a front whose pre-shock side expands raises Gammabar: no threshold
    expanding = fold_of(front(4.0), U=0.3)
    assert (expanding.threshold, expanding.ratio) == (math.inf, 0.0)


def test_the_pre_shock_state_is_read_at_the_first_face_clear_of_the_front():
    # Gammabar / |U| rising outward, as ahead of an outward-facing shock in a trapped region: the threshold is the one
    # at the outer face of the first cell outside the detected run, not further out, where it is larger
    Gammabar2 = (0.02 * np.arange(FOLD_N + 1.0)) ** 2
    for rho, p in ((front(4.0, 2.5), 23), (front(3.0), 22)):
        fold = fold_numbers(rho, np.full(FOLD_N + 1, -1.0), Gammabar2, FOLD_GEO, FOLD_LAYOUT, 3.0, 1.0 / 3.0)
        assert fold.threshold == pytest.approx(0.02 * p, rel=1e-15)


def test_the_fold_monitor_aborts_at_and_above_the_threshold_and_not_below_it():
    # The sharp front of compression 3 has v_12 = 1/sqrt(5); the threshold Gammabar_1/|U_1| is set just either side.
    v12 = taub_velocity(3.0, 1.0 / 3.0)
    below = fold_of(front(3.0), Gammabar=v12 * (1.0 + 1e-9))
    assert below.ratio < 1.0
    check_fold(below)
    for Gammabar in (v12 * (1.0 - 1e-9), v12):  # at the threshold itself Gammabar_2 = 0: the fold begins
        above = fold_of(front(3.0), Gammabar=Gammabar)
        assert above.ratio >= 1.0
        with pytest.raises(ExcisionError, match=r"fold monitor.*excise further out") as failure:
            check_fold(above)
        assert (failure.value.face, failure.value.value) == (19, above.ratio)
    # a measured ratio above one without a detected front is not a fold: only a detected front aborts
    check_fold(dataclasses.replace(below, fronts=0, ratio=2.0))
    # of two fronts, the one nearer the threshold is reported
    two = front(4.0, 2.5)
    two[FOLD_JE:12] = 12.0  # a second front, of compression 3, at face 12, detected in cells 11 and 12
    U = np.full(FOLD_N + 1, -1.0)
    U[14] = -10.0  # the inner front's pre-shock face, the outer face of cell 13, is deep in the trapped region
    fold = fold_numbers(two, U, np.full(FOLD_N + 1, 0.25), FOLD_GEO, FOLD_LAYOUT, 3.0, 1.0 / 3.0)
    assert (fold.fronts, fold.cell, fold.compression, fold.threshold) == (2, 11, 3.0, 0.05)


def test_the_fold_monitor_refuses_values_that_are_not_finite():
    rho = front(3.0)
    rho[25] = np.nan
    with pytest.raises(ExcisionError, match="a density in the zone is not finite"):
        fold_of(rho)
    with pytest.raises(ExcisionError, match="the radius 2 M_AH is not finite"):
        fold_of(front(3.0), X_zone=math.nan)


def test_the_fold_monitor_sees_nothing_on_excised_frw():
    geo = Geometry.of(*IdentityMap(4.0).radii(0.5, 40))
    layout = Layout(40, j_e=5)
    state = frw_state(geo, 5)
    d = derive(state, geo, Background.at(RAD, 0.5), RAD, StencilWeights.of(geo, layout), THETA)
    fold = fold_monitor(state, d, geo, RAD, layout, 2.0 * math.exp(RAD.alpha_float * 0.5), 0.5)  # 2 M_AH at X = 4
    assert fold.X_zone == pytest.approx(4.0, rel=1e-15)
    assert (fold.fronts, fold.jump, fold.compression) == (0, pytest.approx(1.0, abs=1e-14), pytest.approx(1.0))
    assert fold.ratio == 0.0  # FRW expands: no threshold


# --- re-excision and zone extension, on prescribed trapping functions ---


def synthetic_report(h: np.ndarray, apparent_x: float | None) -> HorizonReport:
    N = h.size - 1
    apparent = (
        Horizon(j=int(apparent_x * N), x=apparent_x, X=4.0 * apparent_x, outer=True) if apparent_x is not None else None
    )
    return HorizonReport(
        h=h,
        trapped_faces=int(np.sum(h < 0.0)),
        horizons=(apparent,) if apparent is not None else (),
        apparent=apparent,
        M_AH=float("nan"),
        residual=float("nan"),
        margin=float("nan"),
        margin_face=-1,
        core_margin=float("nan"),
        core_margin_face=-1,
        outer_face_trapped=False,
    )


def test_re_excision_advances_to_the_target_or_as_far_as_the_stencil_stays_trapped():
    N = 100
    h = np.ones(N + 1)
    h[20:61] = -1.0  # trapped from face 20 to face 60: the horizon is near face 60
    layout = Layout(N, j_e=25)
    assert re_excision_face(synthetic_report(h, 0.60), layout, eta_r=0.7) == 42  # ceil(100 * 0.7 * 0.6)
    assert re_excision_face(synthetic_report(h, 0.60), Layout(N, j_e=42), eta_r=0.7) is None  # already there
    assert re_excision_face(synthetic_report(h, 0.60), layout, eta_r=0.3) is None  # the target is behind the face
    h[45:48] = 1.0  # an untrapped gap in the way: the face stops where three faces are still trapped
    assert re_excision_face(synthetic_report(h, 0.60), layout, eta_r=0.7) == 42
    assert re_excision_face(synthetic_report(h, 0.60), Layout(N, j_e=44), eta_r=0.7) is None
    assert re_excision_face(synthetic_report(h, None), layout, eta_r=0.7) is None
    # a horizon far outside, a new trapped region beyond an untrapped gap: the face jumps across the gap once the
    # target lands inside the new region, and until then goes as far as the old region allows
    h = np.ones(N + 1)
    h[20:31] = -1.0
    h[70:91] = -1.0
    assert re_excision_face(synthetic_report(h, 0.90), layout, eta_r=0.7) == 28  # the target 63 sits in the gap
    assert re_excision_face(synthetic_report(h, 0.90), layout, eta_r=0.8) == 72  # the target 72 is trapped
    # nothing trapped between the face and the target: the face stays
    h = np.ones(N + 1)
    h[90:] = -1.0
    assert re_excision_face(synthetic_report(h, 0.95), Layout(N, j_e=40), eta_r=0.7) is None


def test_a_zone_is_extended_when_the_horizon_reaches_the_fraction_of_its_inner_edge():
    zone = Zone(xi_on=1.0, tau_on=0.3, x_t=0.4, Delta_t=0.1)  # inner edge 0.3
    assert not zone_needs_extension(synthetic_report(np.ones(11), 0.20), (zone,), 0.8)
    assert zone_needs_extension(synthetic_report(np.ones(11), 0.25), (zone,), 0.8)
    assert zone_needs_extension(synthetic_report(np.ones(11), 0.70), (zone,), 0.8)  # a jump past the transition
    assert not zone_needs_extension(synthetic_report(np.ones(11), 0.70), (), 0.8)
    assert not zone_needs_extension(synthetic_report(np.ones(11), None), (zone,), 0.8)


# --- the configuration ---


def test_the_excision_configuration_has_the_switch_and_the_optional_re_excision(tmp_path: Path):
    path = tmp_path / "c.yaml"
    text = "grid: {N: 40, Rtilde_max: 4.0, scale: 2.0}\nevolution: {xi_end: 1.0}\n"
    path.write_text(text + "excision: {enabled: false, eta_r: null}\n")
    config = load(path)
    assert config.excision.enabled is False
    assert config.excision.eta_r is None
    assert ExcisionConfig().enabled is True
    assert ExcisionConfig().eta_r == 0.7
    assert ExcisionConfig().zone_extension_at == 0.8


# --- the driver: an excised collapse end to end, the comparison with the unexcised run, and restarts ---


@pytest.fixture(scope="module")
def excised(tmp_path_factory: pytest.TempPathFactory) -> tuple[RunReader, Path, Path]:
    """The same collapse run with excision on, through formation and a zone extension, to xi = 6.5."""
    directory = tmp_path_factory.mktemp("excised")
    path = directory / "bh.yaml"
    path.write_text(
        "grid: {N: 200, Rtilde_max: 12.0, scale: 3.0}\n"
        "output: {snapshot_spacing: 0.5, snapshot_spacing_after: 0.05}\n"
        "evolution: {xi_end: 6.5}\n"
    )
    A = 0.515 * math.e / 8.0
    args = ["initial", "gaussian", "bh", "--config", str(path), "--A", f"{A:.12g}", "--ell", "2.0"]
    assert main([*args, "--dir", str(directory)]) == 0
    assert main(["run", str(path), "bh", "--dir", str(directory)]) == 0
    return RunReader(RunPaths.of(directory, "bh").evolution), path, directory


def test_an_excised_collapse_runs_through_formation_and_far_beyond_it(excised: tuple[RunReader, Path, Path]):
    reader, _, _ = excised
    end = reader.end
    assert end is not None
    assert (
        end.payload["status"] == "completed"
    )  # unexcised, the same collapse dies a fifth of an e-fold after formation
    kinds = [e.kind for e in reader.events]
    assert kinds[:2] == ["formation", "switch_on"]
    assert kinds.count("re_excision") >= 3
    assert "zone_extension" in kinds
    formation, switch = reader.events[0], reader.events[1]
    assert 4.7 < formation.xi < 4.9
    assert switch.xi == formation.xi  # thrown at first detection here: the three faces were already trapped
    assert switch.payload["j_e"] == math.ceil(200 * switch.payload["x_e"])
    assert switch.payload["x_e"] == pytest.approx(0.7 * math.exp(-0.15) * switch.payload["x_AH_on"])
    assert switch.payload["x_t"] + switch.payload["Delta_t"] < 0.8
    for e in reader.events:
        if e.kind == "re_excision":
            assert e.payload["to"] > e.payload["from"]
            assert (
                e.payload["M_je_after"] >= e.payload["M_je_before"]
            )  # the dropped cells' energy joins the mass inside
    extension = next(e for e in reader.events if e.kind == "zone_extension")
    assert extension.payload["x_t"] - extension.payload["Delta_t"] >= switch.payload["x_t"] + switch.payload["Delta_t"]
    # the face stayed in the certified regime at every excised step
    table = reader.horizon
    j_e = np.asarray(table["j_e"], dtype=np.int64)
    rows = j_e >= 0
    assert rows.sum() > 400
    assert np.all(np.diff(j_e[rows]) >= 0)
    assert np.all(np.asarray(table["mu"], dtype=np.float64)[rows] > 0.0)
    assert np.all(np.asarray(table["Lambda_plus"], dtype=np.float64)[rows] == 0.0)
    assert np.max(np.asarray(table["a_over_Theta"], dtype=np.float64)[rows]) < math.sqrt(1.0 / 3.0)
    for name in ("h_e", "h_e1", "h_e2"):
        assert np.all(np.asarray(table[name], dtype=np.float64)[rows] < 0.0)
    M_AH = np.asarray(table["M_AH"], dtype=np.float64)
    assert np.all(np.diff(M_AH[rows]) >= -1e-12)
    assert 4.0 < M_AH[rows][-1] < 6.0  # the paper's ladder: 5.37 three e-folds on at higher resolution
    ratio = np.asarray(table["zone_ratio"], dtype=np.float64)
    assert np.nanmax(ratio) < 1.0
    # the snapshots after the switch carry the face and the zones
    later = [s for s in reader.snapshots if s.xi > switch.xi]
    assert all(s.j_e > 0 for s in later)
    last = reader.snapshot(later[-1].index)
    assert len(last.zones) == 2
    assert last.xi_form == formation.xi


def test_the_excised_run_agrees_with_the_unexcised_continuation_outside_the_face(excised: tuple[RunReader, Path, Path]):
    # The switch-on snapshot is the unexcised state already on the blend: continued with excision off it runs on the
    # same grid, and dies where the unexcised collapse dies. Outside the face the two agree, the closure's error
    # confined to the first cells at the face and never propagating outward: that is what excision promises.
    reader, path, directory = excised
    switch = reader.events[1]
    index = next(s.index for s in reader.snapshots if s.xi == switch.xi and s.j_e == 0)
    off = directory / "off.yaml"
    off.write_text(path.read_text().replace("xi_end: 6.5", "xi_end: 5.5") + "excision: {enabled: false}\n")
    status = main(["restart", "bh", "off", "--dir", str(directory), "--snapshot", str(index), "--config", str(off)])
    assert status == 2  # aborted: the interior breaks the areal coordinate
    unexcised = RunReader(RunPaths.of(directory, "off").evolution)
    assert unexcised.snapshot(0).zones == reader.snapshot(index).zones
    common = {s.xi: s.index for s in unexcised.snapshots}
    compared = 0
    for s in reader.snapshots:
        if s.xi in common and s.xi > switch.xi and s.j_e > 0:
            a, b = reader.snapshot(s.index), unexcised.snapshot(common[s.xi])
            scale = float(np.max(np.abs(b.delta_E[a.j_e :])))
            at_face = np.max(np.abs(a.delta_E[a.j_e : a.j_e + 2] - b.delta_E[a.j_e : a.j_e + 2])) / scale
            outside = np.max(np.abs(a.delta_E[a.j_e + 16 :] - b.delta_E[a.j_e + 16 :])) / scale
            assert at_face > 1e-5  # the first-order layer at the face
            assert outside < 1e-12  # round-off sixteen cells out
            assert np.max(np.abs(a.delta_U[a.j_e + 16 :] - b.delta_U[a.j_e + 16 :])) < 1e-10
            compared += 1
    assert compared >= 3


def test_a_restart_from_an_excised_snapshot_reproduces_the_run_bit_for_bit(excised: tuple[RunReader, Path, Path]):
    reader, _, directory = excised
    switch = reader.events[1]
    middle = next(s for s in reader.snapshots if s.j_e > 0 and s.xi > switch.xi + 0.5)
    assert main(["restart", "bh", "again", "--dir", str(directory), "--snapshot", str(middle.index)]) == 0
    again = RunReader(RunPaths.of(directory, "again").evolution)
    assert again.snapshot(0).j_e == middle.j_e
    final = reader.snapshot(len(reader.snapshots) - 1)
    final_again = again.snapshot(len(again.snapshots) - 1)
    assert final.xi == final_again.xi == 6.5
    assert final.j_e == final_again.j_e
    assert np.array_equal(final.delta_E[final.j_e :], final_again.delta_E[final.j_e :])
    assert np.array_equal(final.delta_U[final.j_e :], final_again.delta_U[final.j_e :])
    assert final.M_e == final_again.M_e
    assert final.zones == final_again.zones
    assert [e.kind for e in again.events if e.kind != "end"] == [
        e.kind for e in reader.events if e.step > middle.step and e.kind != "end"
    ]


def test_an_excised_state_is_refused_by_a_configuration_that_disables_excision(
    excised: tuple[RunReader, Path, Path], capsys: pytest.CaptureFixture[str]
):
    # continuing an excised state without excision would still re-excise and pin zones: refused before anything is run
    reader, path, directory = excised
    disabled = directory / "disabled.yaml"
    disabled.write_text(path.read_text() + "excision: {enabled: false}\n")
    index = excised_index(reader)
    args = ["restart", "bh", "refused", "--dir", str(directory), "--snapshot", str(index), "--config", str(disabled)]
    assert main(args) == 1
    assert "the initial state is excised" in capsys.readouterr().err
    assert not RunPaths.of(directory, "refused").evolution.exists()


def test_a_restart_from_the_switch_on_snapshot_throws_the_switch_again_with_the_stored_zone(
    excised: tuple[RunReader, Path, Path],
):
    reader, path, directory = excised
    switch = reader.events[1]
    index = next(s.index for s in reader.snapshots if s.xi == switch.xi and s.j_e == 0)
    short = directory / "short.yaml"
    short.write_text(path.read_text().replace("xi_end: 6.5", "xi_end: 5.2"))
    assert (
        main(["restart", "bh", "resw", "--dir", str(directory), "--snapshot", str(index), "--config", str(short)]) == 0
    )
    resumed = RunReader(RunPaths.of(directory, "resw").evolution)
    assert next(e.kind for e in resumed.events) == "switch_on"  # at the examination before the first step
    assert resumed.events[0].payload["j_e"] == switch.payload["j_e"]
    assert resumed.snapshot(1).zones == reader.snapshot(index).zones  # the stored zone, not a second one
    assert len(resumed.snapshot(1).zones) == 1
    common = {s.xi: s.index for s in resumed.snapshots if s.j_e > 0}
    matched = [(s.index, common[s.xi]) for s in reader.snapshots if s.xi in common]
    assert matched
    i, k = matched[-1]
    a, b = reader.snapshot(i), resumed.snapshot(k)
    assert np.array_equal(a.delta_E[a.j_e :], b.delta_E[b.j_e :])
    assert a.j_e == b.j_e


# --- the paths a natural run does not reach: forced through the driver's seams ---


def restart_short(directory: Path, path: Path, name: str, index: int, xi_end: float) -> RunReader:
    """Restart the excised run's snapshot `index` as `name`, to `xi_end`, and read it back."""
    short = directory / f"{name}.yaml"
    short.write_text(path.read_text().replace("xi_end: 6.5", f"xi_end: {xi_end}"))
    main(["restart", "bh", name, "--dir", str(directory), "--snapshot", str(index), "--config", str(short)])
    return RunReader(RunPaths.of(directory, name).evolution)


def always(report: HorizonReport, zones: tuple[Zone, ...], fraction: float) -> bool:
    """A zone-extension trigger that always fires."""
    return True


def excised_index(reader: RunReader) -> int:
    switch = reader.events[1]
    return next(s.index for s in reader.snapshots if s.j_e > 0 and s.xi > switch.xi + 0.3)


def test_a_refused_zone_extension_is_logged_once_per_change_of_reasons(
    excised: tuple[RunReader, Path, Path], monkeypatch: pytest.MonkeyPatch
):
    import pbh.driver as driver_module

    reader, path, directory = excised
    monkeypatch.setattr(driver_module, "zone_needs_extension", always)
    real = driver_module.attempt_switch_on

    def refusing(*args: Any, **kwargs: Any) -> SwitchAttempt:
        return dataclasses.replace(real(*args, **kwargs), no_overlap=False)

    monkeypatch.setattr(driver_module, "attempt_switch_on", refusing)
    index = excised_index(reader)
    resumed = restart_short(directory, path, "refused", index, reader.snapshot(index).xi + 0.05)
    attempts = [e for e in resumed.events if e.kind == "zone_attempt"]
    assert len(attempts) == 1  # refused at every step, logged once
    assert attempts[0].payload["failed"] == ["no_overlap"]


def test_a_transition_that_cannot_fit_ends_the_run_at_once_naming_the_radius_it_needs(
    excised: tuple[RunReader, Path, Path], monkeypatch: pytest.MonkeyPatch
):
    # The horizon only grows, so a zone that does not fit now never will: the run ends instead of refusing forever.
    import pbh.driver as driver_module

    reader, path, directory = excised
    monkeypatch.setattr(driver_module, "zone_needs_extension", always)
    real = driver_module.attempt_switch_on

    def too_big(*args: Any, **kwargs: Any) -> SwitchAttempt:
        return dataclasses.replace(real(*args, **kwargs), transition_fits=False)

    monkeypatch.setattr(driver_module, "attempt_switch_on", too_big)
    index = excised_index(reader)
    resumed = restart_short(directory, path, "no_room", index, reader.snapshot(index).xi + 0.05)
    assert [e.kind for e in resumed.events][-2:] == ["abort", "end"]
    abort = next(e for e in resumed.events if e.kind == "abort")
    assert abort.payload["field"] == "transition_fits"
    end = resumed.end
    assert end is not None
    assert "enlarge Rtilde_max" in end.payload["reason"]


def test_the_transition_spans_the_areal_radii_and_on_the_uniform_map_the_labels_are_the_old_rule():
    # On the sinh stretch the labels near the origin are inflated, and a rule in labels would refuse a hole that fits:
    # the transition is placed by areal radius, (c_t -+ c_Delta) X_AH, through the map.
    for grid in (IdentityMap(12.0), SinhStretch(12.0, scale=3.0)):
        X_AH = 1.0
        x_in, x_out = (label_of(grid, 0.0, (EXCISION.c_t + s * EXCISION.c_Delta) * X_AH) for s in (-1.0, 1.0))
        for x, X in (
            (x_in, (EXCISION.c_t - EXCISION.c_Delta) * X_AH),
            (x_out, (EXCISION.c_t + EXCISION.c_Delta) * X_AH),
        ):
            assert float(grid.radius_at(0.0, np.array([x]))[0]) == pytest.approx(X, rel=1e-14)
    x_AH = 1.0 / 12.0  # the uniform map: X = 12 x, so the areal rule is the label rule c_t x_AH, c_Delta x_AH
    x_in, x_out = (label_of(IdentityMap(12.0), 0.0, (EXCISION.c_t + s * EXCISION.c_Delta) * 1.0) for s in (-1.0, 1.0))
    assert 0.5 * (x_in + x_out) == pytest.approx(EXCISION.c_t * x_AH, rel=1e-14)
    assert 0.5 * (x_out - x_in) == pytest.approx(EXCISION.c_Delta * x_AH, rel=1e-14)
    assert label_of(IdentityMap(12.0), 0.0, 100.0) == 2.0  # beyond the grid: enough to fail the fit


def test_a_zone_extension_can_move_the_face_out_to_the_new_horizon(
    excised: tuple[RunReader, Path, Path], monkeypatch: pytest.MonkeyPatch
):
    # An engulfing horizon far outside would give the attempt a face further out than the current one.
    import pbh.driver as driver_module

    reader, path, directory = excised
    index = excised_index(reader)
    record = reader.snapshot(index)
    monkeypatch.setattr(driver_module, "zone_needs_extension", always)
    real = driver_module.attempt_switch_on

    def further(*args: Any, **kwargs: Any) -> SwitchAttempt:
        attempt = dataclasses.replace(real(*args, **kwargs), j_e=record.j_e + 1, x_t=0.55, Delta_t=0.05)
        return dataclasses.replace(attempt, transition_fits=True, no_overlap=True)

    monkeypatch.setattr(driver_module, "attempt_switch_on", further)
    resumed = restart_short(directory, path, "moved", index, record.xi + 0.02)
    kinds = [e.kind for e in resumed.events]
    assert "zone_extension" in kinds
    moved = next(e for e in resumed.events if e.kind == "re_excision" and e.payload["trigger"] == "zone_extension")
    assert moved.payload["to"] == record.j_e + 1


def test_a_failed_face_assertion_aborts_the_run_as_a_result(
    excised: tuple[RunReader, Path, Path], monkeypatch: pytest.MonkeyPatch
):
    import pbh.driver as driver_module

    reader, path, directory = excised
    index = excised_index(reader)

    def failing(*args: Any, **kwargs: Any) -> FaceValues:
        raise ExcisionError("the outflow margin mu > 0", 17, -0.1)

    monkeypatch.setattr(driver_module, "check_face", failing)
    resumed = restart_short(directory, path, "failed", index, reader.snapshot(index).xi + 0.05)
    end = resumed.end
    assert end is not None
    assert end.payload["status"] == "aborted"
    abort = next(e for e in resumed.events if e.kind == "abort")
    assert abort.payload == {"field": "the outflow margin mu > 0", "index": 17, "value": -0.1}


def test_every_excised_step_records_the_fold_monitor_and_the_summary_reports_it(
    excised: tuple[RunReader, Path, Path],
):
    # Every excised row carries an evaluation, and no unexcised row does. The collapse captures no shock inside the
    # horizon, but at N = 200 the switch-on leaves only four or five cells between the face and the horizon, and the
    # smooth infall across them reads above the detection level for a fifth of an e-fold: fronts are detected there,
    # measured across the gradient, and none comes near a fold. Resolution, not a shock; the ratio decides.
    reader = excised[0]
    table = reader.horizon
    rows = np.asarray(table["j_e"], dtype=np.int64) >= 0
    xi = np.asarray(table["xi"], dtype=np.float64)
    fronts = np.asarray(table["fold_fronts"], dtype=np.int64)
    ratio = np.asarray(table["fold_ratio"], dtype=np.float64)
    for name in ("fold_X_zone", "fold_jump", "fold_X", "fold_compression", "fold_U1", "fold_Gammabar1", "fold_v12"):
        column = np.asarray(table[name], dtype=np.float64)
        assert np.all(np.isnan(column[~rows]))
        assert np.all(np.isfinite(column[rows]))
    assert np.all(fronts[~rows] == 0)
    assert np.all(np.asarray(table["fold_cell"], dtype=np.int64)[rows] > np.asarray(table["j_e"])[rows])
    detected = rows & (fronts > 0)
    switch_on = next(e.xi for e in reader.events if e.kind == "switch_on")
    assert np.sum(detected) > 0
    assert np.all(xi[detected] < switch_on + 0.2)
    assert np.all(np.asarray(table["faces_to_horizon"], dtype=np.int64)[detected] <= 5)
    assert np.all((0.0 <= ratio[rows]) & (ratio[rows] < 0.7))
    # the summary: the evaluations, the steps with a front, and the nearest approach to a fold
    fold = summarise(reader).fold
    assert fold is not None
    assert (fold.steps, fold.steps_with_front) == (int(np.sum(rows)), int(np.sum(detected)))
    assert fold.jump_max == float(np.max(np.asarray(table["fold_jump"], dtype=np.float64)[rows]))
    k = int(np.argmax(np.where(rows, ratio, -np.inf)))
    assert (fold.ratio_max, fold.ratio_xi, fold.ratio_cell) == (ratio[k], xi[k], int(table["fold_cell"][k]))
    assert fold.ratio_detected == bool(fronts[k] > 0)
    text = describe(summarise(reader))
    assert f"fold monitor: {fold.steps} excised steps" in text
    assert f"a front detected on {fold.steps_with_front}" in text
    assert as_json(summarise(reader))["fold"]["steps"] == fold.steps


def test_a_fold_aborts_the_run_with_the_row_written_naming_the_fold_and_the_cure(
    excised: tuple[RunReader, Path, Path], monkeypatch: pytest.MonkeyPatch
):
    import pbh.excision as excision_module

    reader, path, directory = excised
    index = excised_index(reader)
    folded = FoldValues(
        X_zone=0.5,
        fronts=1,
        jump=2.0,
        cell=23,
        X=0.4,
        compression=6.0,
        U1=-1.0,
        Gammabar1=0.6,
        v12=0.66,
        threshold=0.6,
        ratio=1.1,
    )

    def folding(*args: Any, **kwargs: Any) -> FoldValues:
        return folded

    monkeypatch.setattr(excision_module, "fold_monitor", folding)
    resumed = restart_short(directory, path, "folded", index, reader.snapshot(index).xi + 0.05)
    end = resumed.end
    assert end is not None
    assert end.payload["status"] == "aborted"
    assert "the slice folds behind a shock of compression 6" in end.payload["reason"]
    assert "excise further out" in end.payload["reason"]
    abort = next(e for e in resumed.events if e.kind == "abort")
    field = "the fold monitor v_12 < Gammabar_1/|Utilde_1|"
    assert abort.payload == {"field": field, "index": 23, "value": 1.1, "fold": dataclasses.asdict(folded)}
    # the evaluation that ended the run is the last row of the horizon table, at the abort's step
    table = resumed.horizon
    assert (int(table["step"][-1]), float(table["xi"][-1])) == (abort.step, abort.xi)
    assert (int(table["fold_cell"][-1]), float(table["fold_ratio"][-1])) == (23, 1.1)


def test_an_excised_step_without_a_trapped_face_ends_the_run_at_the_face_assertion(
    excised: tuple[RunReader, Path, Path], monkeypatch: pytest.MonkeyPatch
):
    # A finder that reports no trapped face once excised: the step is not waved through as an unexcised one, but goes
    # to the face assertions, which fail at the face, no longer trapped.
    import pbh.driver as driver_module

    reader, path, directory = excised
    real = driver_module.find_horizons

    def untrapped(*args: Any, **kwargs: Any) -> HorizonReport:
        report = real(*args, **kwargs)
        h = np.where(np.isnan(report.h), np.nan, np.abs(report.h) + 1.0)
        return dataclasses.replace(report, h=h, trapped_faces=0, horizons=(), apparent=None, M_AH=math.nan)

    monkeypatch.setattr(driver_module, "find_horizons", untrapped)
    index = excised_index(reader)
    resumed = restart_short(directory, path, "untrapped", index, reader.snapshot(index).xi + 0.05)
    end = resumed.end
    assert end is not None
    assert end.payload["status"] == "aborted"
    abort = next(e for e in resumed.events if e.kind == "abort")
    assert abort.payload["field"] == "the faces j_e .. j_e + 2 trapped"
    assert abort.payload["index"] == reader.snapshot(index).j_e


# --- the mass read out of the excised collapse (Section 8.5) ---


@pytest.mark.slow
def test_the_excised_collapse_reads_its_mass_stops_and_a_restart_reads_the_same(tmp_path: Path):
    path = tmp_path / "bh.yaml"
    path.write_text(
        "grid: {N: 200, Rtilde_max: 12.0, scale: 3.0}\n"
        "output: {snapshot_spacing: 0.5, snapshot_spacing_after: 0.05}\n"
        "evolution: {xi_end: 9.0}\n"
    )
    A = 0.515 * math.e / 8.0
    args = ["initial", "gaussian", "bh", "--config", str(path), "--A", f"{A:.12g}", "--ell", "2.0"]
    assert main([*args, "--dir", str(tmp_path)]) == 0
    assert main(["run", str(path), "bh", "--dir", str(tmp_path)]) == 0
    reader = RunReader(RunPaths.of(tmp_path, "bh").evolution)
    events = reader.events
    formation = next(e for e in events if e.kind == "formation").xi
    (readout,) = [e for e in events if e.kind == "readout"]
    p = readout.payload
    # read at the floor or later with its bar below one per cent, and the run stopped there rather than at xi_end
    assert p["xi_reading"] >= formation + 2.0
    assert p["bar"] < 0.01
    assert p["M_est"] == pytest.approx(5.51, abs=0.02)  # the horizon mass is still growing from 5.2 there
    assert p["M_AH"] < p["M_est"]
    assert p["systematic"] < 1e-4
    assert not p["efficiency_flag"]  # within 20 per cent of Michel's by then
    assert reader.end is not None
    assert reader.end.payload["reason"] == "the mass was read"
    assert reader.end.xi < p["xi_reading"] + 0.15 + 2 * READOUT_CHECK + 0.05
    # the near zone at the horizon and at the sonic point is near its steady values, the face an outflow boundary
    near = p["near_zone"]
    assert near["v"][0] == pytest.approx(-1.0, abs=1e-2)  # the finder's definition of the horizon
    assert near["lapse"] == pytest.approx(near["michel_lapse"], rel=0.1)
    assert near["rho"] == pytest.approx(near["michel_rho"], rel=0.3)
    assert p["outflow_margin"] > 0.0
    assert 0.0 < p["min_lapse"] < 1.0
    # a restart from a snapshot half way through the epoch is handed its history and reads the same mass
    index = next(s.index for s in reader.snapshots if s.xi > formation + 1.0)
    assert main(["restart", "bh", "again", "--dir", str(tmp_path), "--snapshot", str(index)]) == 0
    (again,) = [e for e in RunReader(RunPaths.of(tmp_path, "again").evolution).events if e.kind == "readout"]
    assert again.payload == p

"""Tests of pbh.excision on hand-built states and on a real collapse: the switch-on tests, excising, the
assertions, re-excision, and the packing of an excised deviation."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from pbh.cli import main
from pbh.config import ExcisionConfig, RunConfig, load
from pbh.derived import derive
from pbh.driver import RunPaths, run
from pbh.eos import RADIATION, Background, EquationOfState
from pbh.excision import (
    OUTER_STATIC_LABEL,
    ExcisionError,
    SwitchAttempt,
    attempt_switch_on,
    check_face,
    excise,
    outflow_margin,
    packed_deviation,
    re_excision_face,
    zone_needs_extension,
)
from pbh.geometry import Geometry
from pbh.horizon import UNEXCISED, FaceValues, Horizon, HorizonReport, HorizonRow, find_horizons
from pbh.layout import Layout
from pbh.maps import IdentityMap, Zone
from pbh.output import RunReader
from pbh.records import read_initial
from pbh.state import frw_state
from pbh.stencils import FaceClosure, StencilWeights
from pbh.timestep import Scheme

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
    d = derive(state, frame.geo, frame.bg, RAD, StencilWeights.of(frame.geo, layout, FaceClosure.FIRST_ORDER))
    report = find_horizons(state, d, frame.geo, frame.bg, RAD, sch.map, layout, record.xi)
    return record, state, d, frame.geo, frame.bg, report, layout


def scheme_on(config: RunConfig, layout: Layout) -> Scheme:
    """The configuration's scheme on an excised layout."""
    return Scheme(
        RAD, config.grid.build(), layout, FaceClosure.FIRST_ORDER, config.outer.build(), config.shocks.build()
    )


def formed_snapshots(reader: RunReader, config: RunConfig) -> list[int]:
    """The indices of the snapshots with a trapped face."""
    return [s.index for s in reader.snapshots if slice_of(reader, config, s.index)[5].trapped_faces > 0]


# --- the outflow margin ---


def test_frw_faces_are_not_outflow_faces_and_trapped_faces_are():
    geo = Geometry.of(*IdentityMap(4.0).radii(0.5, 40))
    bg = Background.at(RAD, 0.5)
    state = frw_state(geo)
    layout = Layout(40)
    d = derive(state, geo, bg, RAD, StencilWeights.of(geo, layout, FaceClosure.FIRST_ORDER))
    # mu = alpha [X - <ephi> (U + Gammabar)] with U = X and ephi = 1 on FRW: -alpha Gammabar < 0, light escapes
    for j in (1, 10, 39):
        assert outflow_margin(j, state, d, geo, RAD, layout) == pytest.approx(-0.5 * math.sqrt(bg.Gammabar2))


def test_the_switch_on_is_refused_while_the_shell_is_thin_and_thrown_once_it_has_thickened(
    collapse: tuple[RunReader, RunConfig],
):
    reader, config = collapse
    formed = formed_snapshots(reader, config)
    assert formed
    _, state, d, geo, _, report, layout = slice_of(reader, config, formed[-1])
    attempt = attempt_switch_on(report, state, d, geo, RAD, layout, EXCISION, ())
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
    refused = attempt_switch_on(thin, state, d, geo, RAD, layout, EXCISION, ())
    assert not refused.passed
    assert refused.failed == ["three_trapped"]
    assert refused.inside_horizon
    assert refused.margin_positive


def test_a_repeated_switch_on_must_move_the_face_outward_and_must_not_overlap_the_zone(
    collapse: tuple[RunReader, RunConfig],
):
    reader, config = collapse
    index = formed_snapshots(reader, config)[-1]
    _, state, d, geo, _, report, layout = slice_of(reader, config, index)
    attempt = attempt_switch_on(report, state, d, geo, RAD, layout, EXCISION, ())
    assert attempt.passed
    # the same slice seen from a run already excised beyond the candidate face: refused, the face cannot move in
    excised_state, excised_layout = excise(state, layout, attempt.j_e + 2)
    again = attempt_switch_on(report, excised_state, d, geo, RAD, excised_layout, EXCISION, ())
    assert not again.inside_horizon
    # and a zone whose transition the new one would overlap
    overlapping = Zone(xi_on=0.0, tau_on=0.3, x_t=attempt.x_t - attempt.Delta_t, Delta_t=0.5 * attempt.Delta_t)
    assert not attempt_switch_on(report, state, d, geo, RAD, layout, EXCISION, (overlapping,)).no_overlap


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
    d_excised = derive(excised, geo, bg, RAD, StencilWeights.of(geo, excised_layout, FaceClosure.FIRST_ORDER))
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
    attempt = attempt_switch_on(report, state, d, geo, RAD, layout, EXCISION, ())
    assert attempt.passed
    excised, excised_layout = excise(state, layout, attempt.j_e)
    sch = Scheme(
        RAD, config.grid.build(), excised_layout, FaceClosure.FIRST_ORDER, config.outer.build(), config.shocks.build()
    )
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
    row = HorizonRow.of(5, record.xi, report_e, None, face)
    assert (row.j_e, row.mu, row.h_e2) == (face.j_e, face.mu, face.h[2])
    assert HorizonRow.of(5, record.xi, report_e, None).j_e == UNEXCISED.j_e == -1
    # a face outside the trapped region fails the trapped-stencil assertion ...
    outside, outside_layout = excise(state, layout, attempt.j_star + 3)
    sch_out = Scheme(
        RAD, config.grid.build(), outside_layout, FaceClosure.FIRST_ORDER, config.outer.build(), config.shocks.build()
    )
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
        FaceClosure.FIRST_ORDER,
        __import__("pbh.outer", fromlist=["HeldAtFrw"]).HeldAtFrw(),
        __import__("pbh.kernels", fromlist=["CENTRED_SCHEME"]).CENTRED_SCHEME,
    )
    result = sch.evaluate(0.5, layout.pack(state))
    report = find_horizons(state, result.derived, geo, bg, RAD, sch.map, layout, 0.5)
    with pytest.raises(ExcisionError, match="mu > 0"):
        check_face(report, state, result.derived, result.speeds, 0.0, geo, bg, RAD, layout, 0.5)


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

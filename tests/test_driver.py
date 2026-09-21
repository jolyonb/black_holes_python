"""Tests of pbh.driver: a run from initial data to its files, its schedule, its abort, and a bit-identical restart."""

from pathlib import Path

import numpy as np
import pytest
from modes import J1_ZEROS, mode_state, single_mode

from pbh.config import (
    EvolutionConfig,
    GridConfig,
    MapFamily,
    OuterChoice,
    OuterConfig,
    OutputConfig,
    RunConfig,
    ShockConfig,
)
from pbh.driver import RunPaths, far_zone_radius, run
from pbh.eos import Background
from pbh.geometry import Geometry
from pbh.kernels import Kernels
from pbh.output import RunReader
from pbh.records import StateRecord, read_initial, write_initial

N = 40
CONFIG = RunConfig(
    grid=GridConfig(N=N, Rtilde_max=4.0, map=MapFamily.UNIFORM),
    outer=OuterConfig(closure=OuterChoice.HELD),
    shocks=ShockConfig(kernels=Kernels.CENTRED),
    output=OutputConfig(snapshot_spacing=0.1, flush_every=7),
    evolution=EvolutionConfig(xi_end=0.4),
)


def bessel_initial(paths: RunPaths, config: RunConfig = CONFIG, xi_0: float = 0.0) -> StateRecord:
    """A Bessel mode with a node at the outer face, written as the run's initial file."""
    sch = config.scheme()
    geo = sch.frame(xi_0).geo
    mode = single_mode(J1_ZEROS[0] / config.grid.Rtilde_max, 1e-3)
    state = mode_state(mode, Background.at(sch.eos, xi_0), geo)
    write_initial(paths.initial, StateRecord.of(state, geo.X[: geo.N + 1], xi_0, 0, {"method": "bessel"}))
    return read_initial(paths.initial)


def test_a_run_writes_its_three_files_lands_on_its_schedule_and_keeps_its_books(tmp_path: Path):
    paths = RunPaths.of(tmp_path, "mode")
    result = run(CONFIG, bessel_initial(paths), paths)
    assert result.status == "completed"
    assert result.xi == 0.4
    assert paths.config.exists()
    assert paths.evolution.exists()
    reader = RunReader(paths.evolution)
    assert reader.config == CONFIG
    steps = reader.steps
    xi = steps["xi"]
    assert isinstance(xi, np.ndarray)
    assert len(xi) == result.steps
    assert np.all(np.diff(xi) > 0.0)
    assert xi[-1] == 0.4
    assert [s.xi for s in reader.snapshots] == [k * 0.1 for k in range(5)]  # the schedule's own floats, exactly
    limits = steps["limit"]
    assert isinstance(limits, list)
    assert set(limits) <= {"courant", "cap", "output_clip"}
    assert limits.count("output_clip") == 4  # one landing per snapshot
    residual = np.asarray(steps["bookkeeping_residual"], dtype=np.float64)
    assert np.max(residual) < 1e-12
    far = np.asarray(steps["far_zone_delta_rho"], dtype=np.float64)
    assert np.max(far[~np.isnan(far)]) < 1e-9  # full rows only
    full = ~np.isnan(np.asarray(steps["boundary_energy"], dtype=np.float64))
    assert full.sum() == 4  # the full row at every snapshot step, and nowhere else
    assert set(np.asarray(steps["step"])[full]) == {s.step for s in reader.snapshots[1:]}
    end = reader.end
    assert end is not None
    assert end.payload["status"] == "completed"


def test_the_full_row_every_step_switch_and_the_restart_that_reproduces_the_run_bit_for_bit(tmp_path: Path):
    config = CONFIG.model_copy(
        update={"output": OutputConfig(snapshot_spacing=0.1, flush_every=7, monitor_every_step=True)}
    )
    paths = RunPaths.of(tmp_path, "whole")
    run(config, bessel_initial(paths, config), paths)
    whole = RunReader(paths.evolution)
    assert not np.any(np.isnan(np.asarray(whole.steps["boundary_energy"], dtype=np.float64)))
    # restart from the snapshot at xi = 0.2 as a new run: the same state at 0.4, to the last bit
    middle = whole.snapshot(2)
    assert middle.xi == 2 * 0.1
    again = RunPaths.of(tmp_path, "again")
    write_initial(again.initial, middle)
    result = run(config, read_initial(again.initial), again)
    assert result.status == "completed"
    first, second = whole.snapshot(4), RunReader(again.evolution).snapshot(2)
    assert first.xi == second.xi == 4 * 0.1
    assert np.array_equal(first.delta_E, second.delta_E)
    assert np.array_equal(first.delta_U, second.delta_U)
    assert first.W == second.W
    assert second.provenance["source"] == "again.evolution.h5"


def test_a_non_finite_state_aborts_as_a_result_with_the_last_good_snapshot(tmp_path: Path):
    paths = RunPaths.of(tmp_path, "nan")
    initial = bessel_initial(paths)
    broken = StateRecord(initial.delta_E, initial.delta_U, float("nan"), 0.0, initial.X, 0.0, 0, {})
    config = CONFIG.model_copy(update={"outer": OuterConfig()})  # the outgoing-wave closure reads W
    result = run(config, broken, paths)
    assert result.status == "aborted"
    assert result.steps == 0
    reader = RunReader(paths.evolution)
    kinds = [e.kind for e in reader.events]
    assert kinds == ["abort", "end"]
    assert reader.events[0].payload["field"] in ("rho", "Gammabar2", "state")  # NaN fails whichever check is first
    end = reader.end
    assert end is not None
    assert end.payload["status"] == "aborted"
    assert len(reader.snapshots) == 2  # the initial state and the last good state, the same here


def test_data_off_the_grid_or_after_the_end_are_refused(tmp_path: Path):
    paths = RunPaths.of(tmp_path, "bad")
    initial = bessel_initial(paths)
    other = CONFIG.model_copy(update={"grid": GridConfig(N=N, Rtilde_max=5.0, map=MapFamily.UNIFORM)})
    with pytest.raises(ValueError, match="not sampled on the grid"):
        run(other, initial, paths)
    ended = CONFIG.model_copy(update={"evolution": EvolutionConfig(xi_end=0.0)})
    with pytest.raises(ValueError, match="not before the end"):
        run(ended, initial, paths)


def test_the_far_zone_radius_is_the_first_face_beyond_the_perturbation(tmp_path: Path):
    paths = RunPaths.of(tmp_path, "far")
    initial = bessel_initial(paths)
    geo = CONFIG.scheme().frame(0.0).geo
    X = geo.X[: N + 1]
    assert far_zone_radius(initial) == X[N]  # the Bessel mode fills the box
    frw = StateRecord(np.zeros(N), np.zeros(N + 1), 0.0, 0.0, X, 0.0, 0, {})
    assert far_zone_radius(frw) == X[1]  # nothing is perturbed: the far zone starts at once
    delta_E = np.zeros(N)
    delta_E[10] = 0.01 * geo.dV[10]
    local = StateRecord(delta_E, np.zeros(N + 1), 0.0, 0.0, X, 0.0, 0, {})
    assert far_zone_radius(local) == X[12]  # cell 10 perturbed: face 11 bounds it, the zone starts at 12


def test_a_run_with_many_steps_flushes_on_its_cadence_and_can_be_read_while_running(tmp_path: Path):
    config = CONFIG.model_copy(update={"output": OutputConfig(snapshot_spacing=0.2, flush_every=3)})
    paths = RunPaths.of(tmp_path, "flushed")
    result = run(config, bessel_initial(paths, config), paths)
    assert result.steps > 3
    assert len(RunReader(paths.evolution).steps["step"]) == result.steps


def test_a_step_that_produces_a_non_finite_state_aborts_by_the_finiteness_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The stages are fine but the combination is not: patch the stepper to hand back a NaN deviation.
    import pbh.driver as driver_module
    from pbh.timestep import advance_with_stages

    def poisoned(*args: object, **kwargs: object):
        dy_new, stages = advance_with_stages(*args, **kwargs)  # type: ignore[arg-type]
        dy_new[0] = float("nan")
        return dy_new, stages

    monkeypatch.setattr(driver_module, "advance_with_stages", poisoned)
    paths = RunPaths.of(tmp_path, "poison")
    result = run(CONFIG, bessel_initial(paths), paths)
    assert result.status == "aborted"
    reader = RunReader(paths.evolution)
    payload = reader.events[0].payload
    assert (payload["field"], payload["index"]) == ("state", -1)
    assert np.isnan(payload["value"])  # JSON carries NaN


def test_a_run_started_from_a_record_with_zones_runs_on_the_blend_and_keeps_the_formation_time(tmp_path: Path):
    # Until the finder exists a record with zones comes only from a hand-made one; the driver must still build the
    # blend from it and schedule its snapshots from the carried formation time.
    from pbh.maps import BlendMap, Zone
    from pbh.output import next_snapshot_time

    zone = Zone(xi_on=0.1, tau_on=0.3, x_t=0.4, Delta_t=0.1)
    config = CONFIG.model_copy(update={"evolution": EvolutionConfig(xi_end=0.35)})
    blend = BlendMap(config.grid.build(), float(config.fluid.build().alpha), (zone,))
    geo = Geometry.of(*blend.radii(0.1, N))
    X = geo.X[: N + 1]
    record = StateRecord(np.zeros(N), np.zeros(N + 1), 0.0, 0.0, X, 0.1, 0, {}, xi_form=0.05, zones=(zone,))
    paths = RunPaths.of(tmp_path, "zoned")
    write_initial(paths.initial, record)
    result = run(config, read_initial(paths.initial), paths)
    assert result.status == "completed"
    reader = RunReader(paths.evolution)
    times = [s.xi for s in reader.snapshots]
    expected = [0.1]
    while expected[-1] < 0.35:
        expected.append(min(next_snapshot_time(expected[-1], 0.05, config.output), 0.35))
    assert times == expected  # the post-formation schedule, from the carried xi_form
    last = reader.snapshot(len(times) - 1)
    assert last.zones == (zone,)
    assert last.xi_form == 0.05
    assert np.array_equal(last.X, Geometry.of(*blend.radii(0.35, N)).X[: N + 1])  # the moving grid
    assert np.max(np.abs(last.delta_E)) < 1e-12  # FRW stays FRW through the ramp on the blend

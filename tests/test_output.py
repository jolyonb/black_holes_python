"""Tests of pbh.output: the table, the run writer and the run reader, complete and mid-run."""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pbh import h5
from pbh.config import EvolutionConfig, GridConfig, OutputConfig, RunConfig
from pbh.geometry import Geometry
from pbh.layout import Layout
from pbh.maps import BlendMap, Zone
from pbh.output import Event, EventRow, RunReader, RunWriter, StepRow, Table, next_snapshot_time, read_table
from pbh.state import State, frw_state
from pbh.types import FloatArray

CONFIG = RunConfig(grid=GridConfig(N=40, Rtilde_max=4.0, scale=2.0), evolution=EvolutionConfig(xi_end=1.0))


@dataclass(frozen=True)
class MonitoredStep(StepRow):
    """A step row extended the way the monitors will extend it: a scalar and a field."""

    rho_0: float
    margin: FloatArray


# --- the table ---


def test_a_table_holds_every_column_kind_and_shows_only_what_was_flushed(tmp_path: Path):
    with h5.create_file(tmp_path / "t.h5") as f:
        table = Table(h5.create_group(f, "steps"), MonitoredStep, widths={"margin": 3})
        table.append(
            MonitoredStep(step=0, xi=0.0, dxi=0.1, limit="courant", rho_0=1.0, margin=np.array([1.0, 2.0, 3.0]))
        )
        table.append(MonitoredStep(step=1, xi=0.1, dxi=0.1, limit="cap", rho_0=1.5, margin=np.array([4.0, 5.0, 6.0])))
        assert len(read_table(h5.subgroup(f, "steps"))["step"]) == 0
        table.flush()
        table.append(MonitoredStep(step=2, xi=0.2, dxi=0.1, limit="cap", rho_0=2.0, margin=np.zeros(3)))
        t = read_table(h5.subgroup(f, "steps"))
        steps = t["step"]
        assert isinstance(steps, np.ndarray)
        assert steps.dtype == np.int64
        assert list(steps) == [0, 1]
        assert t["xi"] == pytest.approx([0.0, 0.1])
        assert t["limit"] == ["courant", "cap"]
        assert t["rho_0"] == pytest.approx([1.0, 1.5])
        assert np.array_equal(t["margin"], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        table.flush()
        table.flush()  # nothing pending: a no-op
        assert list(read_table(h5.subgroup(f, "steps"))["step"]) == [0, 1, 2]


def test_a_table_refuses_an_array_column_without_its_width_and_an_unknown_column_type(tmp_path: Path):
    @dataclass(frozen=True)
    class Odd:
        flag: bool

    with h5.create_file(tmp_path / "t.h5") as f:
        with pytest.raises(ValueError, match="needs its width"):
            Table(h5.create_group(f, "a"), MonitoredStep)
        with pytest.raises(TypeError, match="must be float, int, str or FloatArray"):
            Table(h5.create_group(f, "b"), Odd)


# --- the writer and the reader ---


def test_the_run_file_records_its_identity_the_steps_the_events_and_the_end(tmp_path: Path):
    path = tmp_path / "run.evolution.h5"
    with RunWriter(path, CONFIG, N=40, row_type=StepRow) as out:
        for i in range(5):
            out.step(StepRow(step=i, xi=0.1 * i, dxi=0.1, limit="courant"))
        out.event(3, 0.3, "formation", {"j_star": 12, "xi_form": 0.3})
        out.close(4, 0.4, "completed", steps=5)
    run = RunReader(path)
    assert run.config == CONFIG
    assert run.N == 40
    assert run.code_commit
    assert run.written.endswith("+00:00")
    assert run.host
    assert list(run.steps["step"]) == [0, 1, 2, 3, 4]
    assert run.steps["limit"] == ["courant"] * 5
    assert run.events[0] == Event(step=3, xi=0.3, kind="formation", payload={"j_star": 12, "xi_form": 0.3})
    assert run.end == Event(step=4, xi=0.4, kind="end", payload={"status": "completed", "steps": 5})


def failing_run(path: Path) -> None:
    with RunWriter(path, CONFIG, N=40, row_type=StepRow) as out:
        out.step(StepRow(step=7, xi=0.7, dxi=0.1, limit="cap"))
        raise RuntimeError("boom")


def test_leaving_by_an_exception_records_an_interrupted_end_and_leaving_normally_a_completed_one(tmp_path: Path):
    path = tmp_path / "run.evolution.h5"
    with pytest.raises(RuntimeError, match="boom"):
        failing_run(path)
    end = RunReader(path).end
    assert end is not None
    assert (end.step, end.xi, end.payload["status"]) == (7, 0.7, "interrupted")
    assert "boom" in end.payload["error"]
    with RunWriter(tmp_path / "empty.evolution.h5", CONFIG, N=40, row_type=StepRow):
        pass  # no steps at all: the end still exists, at step -1
    end = RunReader(tmp_path / "empty.evolution.h5").end
    assert end is not None
    assert (end.step, end.payload["status"]) == (-1, "completed")
    assert np.isnan(end.xi)


def test_closing_twice_is_harmless_and_an_oversized_payload_is_refused(tmp_path: Path):
    path = tmp_path / "run.evolution.h5"
    with RunWriter(path, CONFIG, N=40, row_type=StepRow) as out:
        with pytest.raises(ValueError, match="does not fit the column"):
            out.event(0, 0.0, "huge", {"text": "x" * 9000})
        out.close(0, 0.0, "completed")
        out.close(0, 0.0, "completed")  # already closed: nothing happens
    assert len(RunReader(path).events) == 1


def test_a_missing_or_non_group_name_is_a_key_error(tmp_path: Path):
    with h5.create_file(tmp_path / "t.h5") as f:
        h5.write_array(f, "x", np.zeros(2))
        with pytest.raises(KeyError, match="is not a group"):
            h5.subgroup(f, "x")


def test_a_foreign_file_is_refused(tmp_path: Path):
    path = tmp_path / "foreign.h5"
    with h5.create_file(path) as f:
        h5.write_text(f, "format", "something else")
        h5.write_int(f, "version", 1)
    with pytest.raises(ValueError, match="not a pbh evolution file"):
        RunReader(path)


def test_the_file_can_be_read_by_another_process_while_it_is_being_written(tmp_path: Path):
    path = tmp_path / "run.evolution.h5"
    with RunWriter(path, CONFIG, N=40, row_type=StepRow) as out:
        for i in range(3):
            out.step(StepRow(step=i, xi=0.1 * i, dxi=0.1, limit="courant"))
        out.flush()
        out.event(2, 0.2, "formation", {})
        script = (
            "import sys; from pathlib import Path; from pbh.output import RunReader; "
            "r = RunReader(Path(sys.argv[1])); print(len(r.steps['step']), len(r.events), r.end is None)"
        )
        seen = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=True)
        assert seen.stdout.split() == ["3", "1", "True"]
    assert RunReader(path).end is not None


def test_event_rows_carry_their_payload_as_json():
    row = EventRow(step=1, xi=0.5, kind="end", payload='{"status": "completed"}')
    assert row.payload == '{"status": "completed"}'


# --- snapshots ---


def test_the_snapshot_schedule_is_uniform_in_xi_before_formation_and_in_physical_time_after():
    output = OutputConfig(snapshot_spacing=0.05, snapshot_spacing_after=0.1)
    assert next_snapshot_time(0.3, None, output) == 7 * 0.05  # the next multiple, computed from its index
    assert next_snapshot_time(0.3, 0.5, output) == 7 * 0.05  # not yet formed
    assert next_snapshot_time(0.32, None, output) == 7 * 0.05
    assert next_snapshot_time(0.0, None, output) == 0.05  # a time already reached counts as passed
    xi_form = 2.0
    xi = xi_form
    times = [xi]
    for _ in range(3):
        xi = next_snapshot_time(xi, xi_form, output)
        times.append(xi)
    physical = np.exp(times) / np.exp(xi_form)  # in Hubble times at formation
    assert np.diff(physical) == pytest.approx([0.1, 0.1, 0.1])
    assert np.all(np.diff(np.diff(times)) < 0.0)  # closer and closer in xi, as the steps are
    assert next_snapshot_time(times[1], xi_form, output) == times[2]  # from any time, the same next time


def test_a_snapshot_round_trips_as_a_restartable_state_on_the_configurations_grid(tmp_path: Path):
    path = tmp_path / "run.evolution.h5"
    N = CONFIG.grid.N
    sch = CONFIG.scheme()
    xi, j_e = 0.7, 3
    layout = Layout(N, j_e=j_e)
    geo = Geometry.of(*BlendMap(CONFIG.grid.build(), float(sch.eos.alpha), (Zone(0.5, 0.3, 0.4, 0.1),)).radii(xi, N))
    rng = np.random.default_rng(1)
    E = geo.dV * (1.0 + 1e-3 * rng.standard_normal(N))
    U = geo.X[: N + 1] * (1.0 + 1e-3 * rng.standard_normal(N + 1))
    E[:j_e] = np.nan
    U[:j_e] = np.nan
    state = State(E=E, U=U, W=0.01, M_e=0.2)
    dy = layout.pack(state) - layout.pack(frw_state(geo, j_e))
    zone = Zone(xi_on=0.5, tau_on=0.3, x_t=0.4, Delta_t=0.1)
    with RunWriter(path, CONFIG, N=N, row_type=StepRow) as out:
        out.snapshot(5, 0.2, Layout(N), np.zeros(Layout(N).size), None, ())  # FRW, unexcised, before formation
        out.snapshot(40, xi, layout, dy, 0.5, (zone,))  # after a switch-on at 0.5: the grid is the blend
    run = RunReader(path)
    listing = run.snapshots
    assert [(s.index, s.step, s.xi, s.j_e) for s in listing] == [(0, 5, 0.2, 0), (1, 40, xi, j_e)]
    record = run.snapshot(1)
    deviation = layout.unpack(dy)
    assert np.array_equal(record.delta_E[j_e:], deviation.E[j_e:])  # the integrator's variables, bit for bit
    assert np.all(np.isnan(record.delta_E[:j_e]))
    assert np.array_equal(record.delta_U[j_e:], deviation.U[j_e:])
    assert record.state.E[j_e:] == pytest.approx(E[j_e:], rel=1e-15)
    assert record.W == pytest.approx(0.01)
    assert record.state.M_e == pytest.approx(0.2)
    assert record.M_e == pytest.approx(0.2 - geo.X[j_e] ** 3)  # the deviation from the FRW excised mass
    assert np.array_equal(record.X, geo.X[: N + 1])  # the blended grid, rebuilt from the stored zones
    assert (record.xi, record.j_e) == (xi, j_e)
    assert record.provenance == {"source": "run.evolution.h5", "step": 40, "snapshot": 1}
    assert record.xi_form == 0.5
    assert record.zones == (zone,)
    frw = run.snapshot(0)
    assert np.array_equal(frw.state.E, run.geometry(0.2).dV)
    assert frw.W == 0.0
    assert frw.xi_form is None
    assert frw.zones == ()


WRITER_UNTIL_KILLED = """
import sys, pathlib, numpy as np
from pbh.config import EvolutionConfig, GridConfig, RunConfig
from pbh.geometry import Geometry
from pbh.layout import Layout
from pbh.maps import BlendMap, Zone
from pbh.output import RunWriter, StepRow
cfg = RunConfig(grid=GridConfig(N=40, Rtilde_max=4.0, scale=2.0), evolution=EvolutionConfig(xi_end=1.0))
out = RunWriter(pathlib.Path(sys.argv[1]), cfg, N=40, row_type=StepRow)
s = 0
while True:
    out.step(StepRow(s, 1e-4 * s, 1e-4, "courant"))
    out.flush()
    if s % 20 == 0:
        out.snapshot(s, 1e-4 * s, Layout(40), np.zeros(Layout(40).size), None, ())
    s += 1
"""


@pytest.mark.slow
def test_a_writer_killed_while_flushing_leaves_a_readable_file_without_an_end(tmp_path: Path):
    # The single-writer mode keeps the file consistent at every flush: what was flushed survives a SIGKILL, and the
    # missing end event is how a reader tells a crashed run from a finished one.
    import signal
    import time

    path = tmp_path / "killed.evolution.h5"
    writer = subprocess.Popen([sys.executable, "-c", WRITER_UNTIL_KILLED, str(path)])
    time.sleep(1.0)
    writer.send_signal(signal.SIGKILL)
    writer.wait()
    run = RunReader(path)
    assert len(run.steps["step"]) > 10
    assert len(run.snapshots) >= 1
    assert run.end is None

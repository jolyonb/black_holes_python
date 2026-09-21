"""Tests of pbh.output: the table, the run writer and the run reader, complete and mid-run."""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pbh import h5
from pbh.config import EvolutionConfig, GridConfig, RunConfig
from pbh.output import Event, EventRow, RunReader, RunWriter, StepRow, Table, read_table
from pbh.types import FloatArray

CONFIG = RunConfig(
    grid=GridConfig(N=40, Rtilde_max=4.0, scale=2.0), evolution=EvolutionConfig(xi_start=0.0, xi_end=1.0)
)


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

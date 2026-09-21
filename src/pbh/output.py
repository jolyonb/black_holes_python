"""The evolution file, `name.evolution.h5`: the third of a run's three files, written as the run goes.

Everything in the file that grows is a `Table`: a group of resizable datasets that share the row axis, one dataset
per column, the columns declared once from a frozen dataclass whose fields are floats, ints, strings or arrays of a
fixed width. Rows are appended into memory and flushed to disk every so many steps and at the end. All tables are
created before the first row is written and the file is then put into HDF5's single-writer multiple-reader mode,
which is what lets another process read the file while the run writes it; the price is that nothing new can be
created afterwards, so every kind of record is a row of a table that exists from the start, including the end of
the run.

    steps      one row per step: step, xi, dxi, which limit set the step, and the monitors of the run
    events     one row per event: step, xi, kind, and a JSON payload (formation, switch-on, abort, end, ...)
    snapshots  one row per output time: the integrator's variables only, from which everything else is derived

A snapshot stores what the integrator carries and nothing derived: the deviations from FRW of the cell energies and
the face velocities, `W`, `M_e`, the excision face and the time. Radii, densities, the lapse and the rest are
recomputed by the reader with the same functions the run used, so that what is plotted is what the run saw, and
every snapshot is a state to restart from. The snapshot times are a schedule that depends only on the
configuration and the formation time, so that two runs of the same collapse, one excised and one not, write their
snapshots at the same times: uniform in `xi` before formation, and after it uniform in physical time in units of
the Hubble time at formation, `e^xi` advancing by `snapshot_spacing_after e^(xi_form)` per snapshot, which is the
clock the hole runs on. The driver clips its steps to land on those times exactly.

The root carries the run's identity as attributes: the format tag and version, the complete text of the
configuration as saved, the code's commit, the time of writing and the host. A reader rebuilds the `Scheme` from
that configuration and computes derived fields with the same functions the run used.

    with RunWriter(path, config, N, row_type=StepRow) as out:
        out.step(StepRow(step=0, xi=0.0, dxi=0.01, limit="courant"))
        out.snapshot(step, xi, layout, dy)              # the packed deviation, as the integrator holds it
        out.event(0, 0.0, "formation", {"j_star": 40})
        ...
    # at close, the end event is written and everything is flushed
    run = RunReader(path)
    run.steps["xi"], run.events, run.end, run.snapshots, run.snapshot(i)   # a StateRecord, restartable
"""

import dataclasses
import json
import platform
import socket
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
import yaml

from pbh import h5
from pbh.config import OutputConfig, RunConfig, code_commit
from pbh.geometry import Geometry
from pbh.h5 import Column
from pbh.layout import Layout
from pbh.records import StateRecord
from pbh.types import FloatArray

FORMAT = "pbh-evolution"
VERSION = 1


# --- tables ---


class Table[R]:
    """A group of resizable datasets sharing the row axis, with rows buffered in memory until flushed.

    The columns come from the fields of a frozen dataclass `R`: `float` and `int` are scalar columns, `str` a string
    column of at most 64 characters unless the field's metadata gives a `length`, and `FloatArray` a column of fixed
    width, the width given at creation. Every dataset is chunked and unlimited along the rows, so appending never
    rewrites what is on disk.
    """

    def __init__(self, group: h5.Group, row_type: type[R], widths: dict[str, int] | None = None) -> None:
        """Create the table's datasets in `group`, empty; `widths` gives the width of each array column."""
        self.group = group
        self.row_type = row_type
        self.fields = [f.name for f in dataclasses.fields(row_type)]  # type: ignore[reportArgumentType]
        self.lengths: dict[str, int] = {}  # the string columns and their capacities
        self.pending: list[R] = []
        for f in dataclasses.fields(row_type):  # type: ignore[reportArgumentType]
            kind = f.type if isinstance(f.type, str) else f.type.__name__
            if kind in ("float", "int"):
                h5.create_column(group, f.name, kind)
            elif kind == "str":
                self.lengths[f.name] = int(f.metadata.get("length", 64))
                h5.create_column(group, f.name, "str", length=self.lengths[f.name])
            elif kind == "FloatArray":
                if widths is None or f.name not in widths:
                    raise ValueError(f"the array column {f.name!r} needs its width")
                h5.create_column(group, f.name, "float", widths[f.name])
            else:
                raise TypeError(f"a table column must be float, int, str or FloatArray, not {kind}")

    def append(self, row: R) -> None:
        """Buffer one row; a string longer than its column holds is refused here, before anything is buffered."""
        for name, length in self.lengths.items():
            size = len(str(getattr(row, name)).encode())
            if size > length:
                raise ValueError(f"a string of {size} bytes does not fit the column {name!r} of {length}")
        self.pending.append(row)

    def flush(self) -> None:
        """Write the buffered rows to disk, where a reader can see them."""
        if not self.pending:
            return
        for name in self.fields:
            h5.append_column(self.group, name, [getattr(row, name) for row in self.pending])
        self.pending.clear()


def read_table(group: h5.Group) -> dict[str, Column]:
    """Every column of a table; a reader sees what has been flushed."""
    return {name: h5.read_column(group, name) for name in h5.column_names(group)}


# --- the rows every run has ---


@dataclass(frozen=True)
class StepRow:
    """The identification of a step; the monitors of a run extend this with their own columns."""

    step: int
    xi: float
    dxi: float
    limit: str
    """Which limit set the step: `courant`, `cap` or `output_clip`."""


@dataclass(frozen=True)
class EventRow:
    """An event: something that happened at a step, with a payload the kind decides."""

    step: int
    xi: float
    kind: str
    payload: str = field(metadata={"length": 8192})
    """A JSON mapping of at most 8 KB."""


@dataclass(frozen=True)
class Event:
    """An event as read back, its payload decoded."""

    step: int
    xi: float
    kind: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class SnapshotRow:
    """A snapshot: the integrator's variables at an output time, and where the excision face is."""

    step: int
    xi: float
    j_e: int
    W: float
    M_e: float
    delta_E: FloatArray
    """`E_c - Delta V_c` on the cells, NaN below the excision face."""
    delta_U: FloatArray
    """`U_j - X_j` on the faces, NaN below the excision face."""


@dataclass(frozen=True)
class SnapshotInfo:
    """What the snapshot listing gives without reading the fields."""

    index: int
    step: int
    xi: float
    j_e: int


def next_snapshot_time(xi: float, xi_form: float | None, output: OutputConfig) -> float:
    """The first snapshot time after `xi`: the schedule both the excised and the unexcised run compute.

    Before formation the snapshot times are the multiples of `snapshot_spacing`, `k snapshot_spacing`; from
    formation on they are the times at which the physical time has advanced from formation by whole multiples of
    `snapshot_spacing_after` Hubble times at formation, `xi_form + ln(1 + m snapshot_spacing_after)`. Each time is
    computed from its index and never by accumulation, so two runs of the same collapse, or a run and its restart,
    land on the same floating-point times; a time already reached counts as passed.
    """
    if xi_form is None or xi < xi_form:
        k = int(np.floor(xi / output.snapshot_spacing + 1e-9)) + 1
        return k * output.snapshot_spacing
    m = int(np.floor((np.exp(xi - xi_form) - 1.0) / output.snapshot_spacing_after + 1e-9)) + 1
    return xi_form + float(np.log1p(m * output.snapshot_spacing_after))


# --- writing ---


class RunWriter[R: StepRow]:
    """Writes the evolution file of one run; use as a context manager so that the end is always recorded.

    Args:
        path: The file to create.
        config: The run's configuration, stored in full.
        N: The number of cells, the width of the field columns.
        row_type: The step row type; `StepRow` or a dataclass extending it with the run's monitors.
    """

    def __init__(self, path: Path, config: RunConfig, N: int, row_type: type[R]) -> None:
        self.file = h5.create_file(path)
        h5.write_text(self.file, "format", FORMAT)
        h5.write_int(self.file, "version", VERSION)
        h5.write_text(self.file, "config", yaml.safe_dump(config.model_dump(mode="json", exclude_none=True)))
        h5.write_text(self.file, "code_commit", code_commit())
        h5.write_text(self.file, "written", datetime.now(UTC).isoformat())
        h5.write_text(self.file, "host", f"{socket.gethostname()} ({platform.platform()})")
        h5.write_int(self.file, "N", N)
        self.steps: Table[R] = Table(h5.create_group(self.file, "steps"), row_type)
        self.events = Table(h5.create_group(self.file, "events"), EventRow)
        widths = {"delta_E": N, "delta_U": N + 1}
        self.snapshots = Table(h5.create_group(self.file, "snapshots"), SnapshotRow, widths)
        h5.start_single_writer_mode(self.file)
        self.closed = False

    def snapshot(self, step: int, xi: float, layout: Layout, dy: FloatArray) -> None:
        """Record a snapshot from the packed deviation `dy` the integrator holds, and flush it at once."""
        deviation = layout.unpack(dy)  # the deviation in the state's shape: delta_E, delta_U, W, M_e
        row = SnapshotRow(
            step=step, xi=xi, j_e=layout.j_e, W=deviation.W, M_e=deviation.M_e, delta_E=deviation.E, delta_U=deviation.U
        )
        self.snapshots.append(row)
        self.snapshots.flush()

    def step(self, row: R) -> None:
        """Record one step."""
        self.steps.append(row)

    def event(self, step: int, xi: float, kind: str, payload: dict[str, Any]) -> None:
        """Record one event; the payload must be JSON-serialisable."""
        self.events.append(EventRow(step=step, xi=xi, kind=kind, payload=json.dumps(payload, sort_keys=True)))
        self.events.flush()  # events are rare and worth seeing at once

    def flush(self) -> None:
        """Write everything buffered to disk."""
        self.steps.flush()
        self.events.flush()

    def close(self, step: int, xi: float, status: str, **payload: object) -> None:
        """Record the end of the run with its status (`completed`, `aborted`, ...) and close the file."""
        if self.closed:
            return
        self.event(step, xi, "end", {"status": status, **payload})
        self.flush()
        self.file.close()
        self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, kind: type[BaseException] | None, value: BaseException | None, tb: TracebackType | None) -> None:
        """Close the file; a run that leaves by an exception is recorded as `interrupted` with the exception's text."""
        if not self.closed:
            last = self.steps.pending[-1] if self.steps.pending else None
            step, xi = (last.step, last.xi) if last is not None else (-1, float("nan"))
            self.close(step, xi, "completed" if value is None else "interrupted", error=repr(value) if value else "")


# --- reading ---


class RunReader:
    """Reads an evolution file, complete or still being written; every property re-reads the file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with self._open() as f:
            if h5.read_text(f, "format") != FORMAT or h5.read_int(f, "version") != VERSION:
                raise ValueError(f"{path} is not a pbh evolution file of a version this code reads")
            self.config = RunConfig.model_validate(yaml.safe_load(h5.read_text(f, "config")))
            self.code_commit = h5.read_text(f, "code_commit")
            self.written = h5.read_text(f, "written")
            self.host = h5.read_text(f, "host")
            self.N = h5.read_int(f, "N")

    def _open(self) -> h5.File:
        return h5.open_file(self.path)

    def _table(self, name: str) -> dict[str, Column]:
        with self._open() as f:
            return read_table(h5.subgroup(f, name))

    @property
    def steps(self) -> dict[str, Column]:
        """The step table, column by column."""
        return self._table("steps")

    @property
    def events(self) -> list[Event]:
        """The events in order."""
        t = self._table("events")
        steps, xis, kinds, payloads = t["step"], t["xi"], t["kind"], t["payload"]
        assert isinstance(kinds, list)
        assert isinstance(payloads, list)
        return [
            Event(step=int(steps[i]), xi=float(xis[i]), kind=kinds[i], payload=json.loads(payloads[i]))
            for i in range(len(kinds))
        ]

    @property
    def end(self) -> Event | None:
        """The end event, or `None` while the run is still going."""
        ends = [e for e in self.events if e.kind == "end"]
        return ends[-1] if ends else None

    @property
    def snapshots(self) -> list[SnapshotInfo]:
        """The snapshots so far: index, step, time and excision face, without reading the fields."""
        with self._open() as f:
            group = h5.subgroup(f, "snapshots")
            steps, xis, faces = (h5.read_column(group, name) for name in ("step", "xi", "j_e"))
        return [SnapshotInfo(i, int(steps[i]), float(xis[i]), int(faces[i])) for i in range(len(steps))]

    def snapshot(self, index: int) -> StateRecord:
        """One snapshot as a state record, on the grid the configuration gives at that time.

        The stored deviation goes into the record untouched, so a run started from it carries the integrator's
        variables to the last bit; the radii come from the configuration's map. The record's provenance names this
        file and the step.
        """
        with self._open() as f:
            group = h5.subgroup(f, "snapshots")

            def column(name: str) -> FloatArray:
                return np.asarray(h5.read_column(group, name), dtype=np.float64)

            step, xi, j_e = int(column("step")[index]), float(column("xi")[index]), int(column("j_e")[index])
            W, M_e = float(column("W")[index]), float(column("M_e")[index])
            delta_E, delta_U = column("delta_E")[index], column("delta_U")[index]
        geo = self.geometry(xi)
        provenance: dict[str, Any] = {"source": self.path.name, "step": step, "snapshot": index}
        return StateRecord(delta_E, delta_U, W, M_e, geo.X[: geo.N + 1], xi, j_e, provenance)

    def geometry(self, xi: float) -> Geometry:
        """The grid at time `xi`, from the configuration's map."""
        return Geometry.of(*self.config.grid.build().radii(xi, self.N))

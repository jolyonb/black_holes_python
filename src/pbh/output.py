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

The root carries the run's identity as attributes: the format tag and version, the complete text of the
configuration as saved, the code's commit, the time of writing and the host. A reader rebuilds the `Scheme` from
that configuration and computes derived fields with the same functions the run used.

    with RunWriter(path, config, N, row_type=StepRow) as out:
        out.step(StepRow(step=0, xi=0.0, dxi=0.01, limit="courant"))
        out.event(0, 0.0, "formation", {"j_star": 40})
        ...
    # at close, the end event is written and everything is flushed
    run = RunReader(path)
    run.steps["xi"], run.events, run.end
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

import yaml

from pbh import h5
from pbh.config import RunConfig, code_commit
from pbh.h5 import Column

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
        h5.start_single_writer_mode(self.file)
        self.closed = False

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

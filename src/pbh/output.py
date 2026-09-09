"""Writers for evolution snapshots.

A snapshot is the set of named quantities (grid index, fields, derived quantities and the time xi) at one output
time. :class:`GnuplotWriter` streams snapshots as tab-separated text blocks; :class:`NpzWriter` collects them and
saves a single ``.npz`` archive with one array per quantity, stacked along a leading snapshot axis.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, Self, TextIO

import numpy as np
from numpy.typing import NDArray

type Snapshot = dict[str, NDArray[Any] | float]
"""Named quantities at one output time: arrays with one value per gridpoint, or scalars (such as xi)."""


class SnapshotWriter(Protocol):
    """Anything that can receive snapshots from :meth:`pbh.base.BlackHoleEvolver.drive`."""

    def write(self, snapshot: Snapshot) -> None:
        """Record one snapshot."""
        ...


class GnuplotWriter:
    """Write snapshots as gnuplot-style text.

    Each snapshot is a block of tab-separated rows (one per gridpoint) with a ``# name name ...`` header line,
    and blocks are separated by a blank line. The file is flushed after every snapshot.
    """

    def __init__(self, file_handle: TextIO) -> None:
        self.file_handle = file_handle

    def write(self, snapshot: Snapshot) -> None:
        """Write one block."""
        self.file_handle.write("# " + "\t".join(snapshot) + "\n")
        gridpoints = max(len(value) for value in snapshot.values() if isinstance(value, np.ndarray))
        for i in range(gridpoints):
            row = [value[i] if isinstance(value, np.ndarray) else value for value in snapshot.values()]
            self.file_handle.write("\t".join(map(str, row)) + "\n")
        self.file_handle.write("\n")
        self.file_handle.flush()


class NpzWriter:
    """Collect snapshots and save them as a single ``.npz`` archive.

    Each quantity becomes an array with a leading snapshot axis, so ``data["rho"][k]`` is the density profile at
    snapshot ``k`` and ``data["xi"][k]`` is its time. The grid index is stored once as a 1D array. Load the result
    with ``numpy.load``.

    An archive cannot be appended to, so the file is written in full after the first snapshot (so that it exists
    early and can be inspected) and again with everything collected on :meth:`save` or on leaving the context,
    including when an exception is propagating. Each write replaces the file atomically.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._snapshots: list[Snapshot] = []

    def write(self, snapshot: Snapshot) -> None:
        """Record one snapshot. The first snapshot is saved immediately; later ones wait for :meth:`save`."""
        self._snapshots.append(snapshot)
        if len(self._snapshots) == 1:
            self.save()

    def save(self) -> None:
        """Write everything collected so far to :attr:`path`."""
        arrays: dict[str, NDArray[Any]] = {}
        if self._snapshots:
            for name in self._snapshots[0]:
                arrays[name] = np.stack([np.asarray(snapshot[name]) for snapshot in self._snapshots])
            if "index" in arrays:
                arrays["index"] = arrays["index"][0]
        # Write to a temporary file (as an open handle, so numpy does not append its own suffix) and rename it
        # into place, so that the archive at self.path is never left half-written.
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        with tmp_path.open("wb") as f:
            # numpy's stub cannot tell **arrays apart from savez's own keyword arguments
            np.savez(f, **arrays)  # pyright: ignore[reportArgumentType]
        os.replace(tmp_path, self.path)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.save()


@contextmanager
def open_writer(path: str | Path) -> Generator[SnapshotWriter]:
    """Open a writer for ``path``, chosen by suffix: ``.npz`` for :class:`NpzWriter`, anything else for gnuplot text."""
    path = Path(path)
    if path.suffix == ".npz":
        with NpzWriter(path) as writer:
            yield writer
    else:
        with path.open("w") as f:
            yield GnuplotWriter(f)

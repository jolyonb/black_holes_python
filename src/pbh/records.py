"""The state record: what an initial-data file and a checkpoint both hold, and the initial-data file itself.

A run is three files, and `name.initial.h5` is the second: the initial data are the initial data, however they were
made. The file holds the state on the grid, the face radii it was sampled on and the time, in the same record that
the evolution file's checkpoints use, so that starting a run from a saved state and continuing one are the same
read. Next to the state sits a free-form provenance mapping, whatever the maker chose to record (the profile and
its amplitude, the spectrum and its seed, the reports of `initial.py`), together with the code's commit and the time
of writing. The code stores the mapping and never interprets it.

    write_initial(path, state, geo, xi, {"method": "growing_mode", "profile": "gaussian", "A": 0.35, "ell": 2.0})
    record = read_initial(path)      # record.state, record.X, record.xi, record.j_e, record.provenance

The layout of a state record in an HDF5 group:

    attrs: format = "pbh-state", version = 1, xi, j_e, W, M_e
    E   (N cells), U (N + 1 faces), X (N + 1 face radii)
    provenance/  attrs: code_commit, written, details (JSON)

The driver checks the record's `X` against the grid its configuration builds before using the state.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pbh import h5
from pbh.config import code_commit
from pbh.geometry import Geometry
from pbh.state import State
from pbh.types import FloatArray

FORMAT = "pbh-state"
VERSION = 1


@dataclass(frozen=True)
class StateRecord:
    """A state with the grid and time it belongs to, and where it came from.

    Attributes:
        state: The state.
        X: The face radii `X_0..X_N` the state is sampled on.
        xi: The time.
        j_e: The excision face; `0` while there is no excision.
        provenance: The free-form mapping the maker recorded, plus `code_commit` and `written`.
    """

    state: State
    X: FloatArray
    xi: float
    j_e: int
    provenance: dict[str, Any]


def write_record(group: h5.Group, state: State, X: FloatArray, xi: float, j_e: int, provenance: dict[str, Any]) -> None:
    """Write a state record into a group: the initial file's root, or a checkpoint's group."""
    if state.E.size + 1 != X.size or state.U.size != X.size:
        raise ValueError(f"the state ({state.E.size} cells) does not fit the grid ({X.size - 1} cells)")
    h5.write_text(group, "format", FORMAT)
    h5.write_int(group, "version", VERSION)
    h5.write_float(group, "xi", xi)
    h5.write_int(group, "j_e", j_e)
    h5.write_float(group, "W", state.W)
    h5.write_float(group, "M_e", state.M_e)
    h5.write_array(group, "E", state.E)
    h5.write_array(group, "U", state.U)
    h5.write_array(group, "X", X)
    made = h5.create_group(group, "provenance")
    h5.write_text(made, "code_commit", code_commit())
    h5.write_text(made, "written", datetime.now(UTC).isoformat())
    h5.write_mapping(made, "details", provenance)


def read_record(group: h5.Group) -> StateRecord:
    """Read a state record from a group written by `write_record`."""
    if h5.read_text(group, "format") != FORMAT or h5.read_int(group, "version") != VERSION:
        raise ValueError("not a pbh state record of a version this code reads")
    state = State(
        E=h5.read_array(group, "E"),
        U=h5.read_array(group, "U"),
        W=h5.read_float(group, "W"),
        M_e=h5.read_float(group, "M_e"),
    )
    made = group["provenance"]
    assert isinstance(made, h5py.Group)
    provenance = h5.read_mapping(made, "details")
    provenance["code_commit"] = h5.read_text(made, "code_commit")
    provenance["written"] = h5.read_text(made, "written")
    return StateRecord(
        state=state,
        X=h5.read_array(group, "X"),
        xi=h5.read_float(group, "xi"),
        j_e=h5.read_int(group, "j_e"),
        provenance=provenance,
    )


def write_initial(path: Path, state: State, geo: Geometry, xi: float, provenance: dict[str, Any]) -> None:
    """Write the initial-data file: the state on the grid `geo` at the time `xi`, with its provenance."""
    with h5py.File(path, "w") as f:
        write_record(f, state, np.asarray(geo.X[: geo.N + 1]), xi, 0, provenance)


def read_initial(path: Path) -> StateRecord:
    """Read an initial-data file."""
    with h5py.File(path, "r") as f:
        return read_record(f)

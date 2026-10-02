"""The state record: what an initial-data file and a checkpoint both hold, and the initial-data file itself.

A run is three files, and `name.initial.h5` is the second: the initial data are the initial data, however they were
made. The file holds the state on the grid, the face radii it was sampled on and the time, in the same record that the
evolution file's snapshots use, so that starting a run from a saved state and continuing one are the same read. The
record stores what the integrator carries, the deviations from FRW of the cell energies and the face velocities, not
the fields themselves: `E - Delta V` keeps the digits that `E` would lose when the deviation is small, and a run
restarted from a snapshot then reproduces the uninterrupted run to the last bit. A cell far below the background is
stored whole by the integrator (`storage.py`), and its record keeps that content itself as well, in `E_whole`, which is
NaN in every other cell, so that a restart from inside a void is bit-exact too. Next to the state sits a free-form
provenance mapping, whatever the maker chose to record (the profile and its amplitude, the spectrum and its seed, the
reports of `initial.py`), together with the code's commit and the time of writing. The code stores the mapping and
never interprets it.

    write_initial(path, StateRecord.of(state, X, xi, 0, {"method": "growing_mode", "profile": "gaussian"}))
    record = read_initial(path)      # .state, .deviation, .X, .xi, .j_e, .provenance

The layout of a state record in an HDF5 group:

    attrs: format = "pbh-state", version = 1, xi, j_e, W, M_e, xi_form (NaN before formation), zones (JSON)
    delta_E (N cells), delta_U (N + 1 faces), X (N + 1 face radii), E_whole (N cells, NaN but in cells stored whole)
    provenance/  attrs: code_commit, written, details (JSON)

The driver checks the record's `X` against the grid its configuration builds before using the state.
"""

import dataclasses
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pbh import h5
from pbh.config import code_commit
from pbh.geometry import shell_volumes
from pbh.maps import Zone
from pbh.state import State
from pbh.storage import any_whole, whole_of
from pbh.types import BoolArray, FloatArray

FORMAT = "pbh-state"
VERSION = 1


@dataclass(frozen=True)
class StateRecord:
    """A state as its deviation from FRW, with the grid and time it belongs to, and where it came from.

    Attributes:
        delta_E: `E_c - Delta V_c` on the cells, NaN below the excision face.
        delta_U: `U_j - X_j` on the faces, NaN below the excision face.
        W: The outer scalar, whose FRW value is zero; `M_e` the excised mass minus its FRW value `X_e^3`.
        X: The face radii `X_0..X_N` the state is sampled on.
        xi: The time.
        j_e: The excision face; `0` while there is no excision.
        provenance: The free-form mapping the maker recorded, plus `code_commit` and `written`.
        xi_form: The formation time, `None` before formation; the snapshot schedule depends on it.
        zones: The pinned zones of the post-formation map (Section 8.1), empty before the first switch-on. With the
            configuration's base map these rebuild the grid the state sits on, so a restart inside or after a
            switch-on, repeated or not, is complete.
        E_whole: The content `E_c` of the cells stored whole (`storage.py`), NaN in every other cell; `None` if none
            is. `delta_E` holds their rounded deviation as well, for a reader that wants only deviations.
    """

    delta_E: FloatArray
    delta_U: FloatArray
    W: float
    M_e: float
    X: FloatArray
    xi: float
    j_e: int
    provenance: dict[str, Any]
    xi_form: float | None = None
    zones: tuple[Zone, ...] = ()
    E_whole: FloatArray | None = None

    @classmethod
    def of(
        cls,
        state: State,
        X: FloatArray,
        xi: float,
        j_e: int,
        provenance: dict[str, Any],
        xi_form: float | None = None,
        zones: tuple[Zone, ...] = (),
    ) -> StateRecord:
        """The record of a state on the faces `X`: the deviation is formed here, once."""
        return cls(
            delta_E=state.E - shell_volumes(X),
            delta_U=state.U - X,
            W=state.W,
            M_e=state.M_e - frw_excised_mass(X, j_e),
            X=X,
            xi=xi,
            j_e=j_e,
            provenance=provenance,
            xi_form=xi_form,
            zones=zones,
        )

    @classmethod
    def of_deviation(cls, deviation: State, X: FloatArray, xi: float, provenance: dict[str, Any]) -> StateRecord:
        """The record of initial data known as their deviation from FRW, before any excision, taken as they are.

        What `pbh.initial` builds: a perturbation below round-off of the background survives only if FRW is never
        added to it on the way, as `of` would.
        """
        return cls(
            delta_E=deviation.E, delta_U=deviation.U, W=deviation.W, M_e=0.0, X=X, xi=xi, j_e=0, provenance=provenance
        )

    @property
    def state(self) -> State:
        """The state itself, FRW plus the deviation, and the content itself in cells stored whole."""
        frw = State(E=shell_volumes(self.X), U=self.X, W=0.0, M_e=frw_excised_mass(self.X, self.j_e))
        state = whole_of(self.stored, self.whole, frw)
        return State(E=state.E, U=state.U, W=self.W, M_e=state.M_e)

    @property
    def deviation(self) -> State:
        """The deviation in the state's shape, rounded in cells stored whole (the integrator's numbers are `stored`)."""
        return State(E=self.delta_E, U=self.delta_U, W=self.W, M_e=self.M_e)

    @property
    def whole(self) -> BoolArray | None:
        """Which cells are stored whole, `None` if none is."""
        return None if self.E_whole is None else any_whole(~np.isnan(self.E_whole))

    @property
    def stored(self) -> State:
        """What the integrator carries, in the state's shape: the deviation, and the content in cells stored whole."""
        whole = self.whole
        if whole is None:
            return self.deviation
        assert self.E_whole is not None
        return State(E=np.where(whole, self.E_whole, self.delta_E), U=self.delta_U, W=self.W, M_e=self.M_e)


def frw_excised_mass(X: FloatArray, j_e: int) -> float:
    """The FRW value of the excised mass, `X_e^3`; zero while there is no excision."""
    return float(X[j_e]) ** 3 if j_e > 0 else 0.0


def write_record(group: h5.Group, record: StateRecord) -> None:
    """Write a state record into a group: the initial file's root, or any other."""
    if record.delta_E.size + 1 != record.X.size or record.delta_U.size != record.X.size:
        raise ValueError(f"the state ({record.delta_E.size} cells) does not fit the grid ({record.X.size - 1} cells)")
    h5.write_text(group, "format", FORMAT)
    h5.write_int(group, "version", VERSION)
    h5.write_float(group, "xi", record.xi)
    h5.write_int(group, "j_e", record.j_e)
    h5.write_float(group, "W", record.W)
    h5.write_float(group, "M_e", record.M_e)
    h5.write_float(group, "xi_form", float("nan") if record.xi_form is None else record.xi_form)
    h5.write_mapping(group, "zones", {"zones": zones_as_mappings(record.zones)})
    h5.write_array(group, "delta_E", record.delta_E)
    h5.write_array(group, "delta_U", record.delta_U)
    h5.write_array(group, "X", record.X)
    h5.write_array(
        group, "E_whole", record.E_whole if record.E_whole is not None else np.full(record.delta_E.size, np.nan)
    )
    made = h5.create_group(group, "provenance")
    h5.write_text(made, "code_commit", code_commit())
    h5.write_text(made, "written", datetime.now(UTC).isoformat())
    h5.write_mapping(made, "details", record.provenance)


def read_record(group: h5.Group) -> StateRecord:
    """Read a state record from a group written by `write_record`.

    A record written before cells were stored whole has no `E_whole`, and stands for one with none stored whole.
    """
    if h5.read_text(group, "format") != FORMAT or h5.read_int(group, "version") != VERSION:
        raise ValueError("not a pbh state record of a version this code reads")
    made = h5.subgroup(group, "provenance")
    provenance = h5.read_mapping(made, "details")
    provenance["code_commit"] = h5.read_text(made, "code_commit")
    provenance["written"] = h5.read_text(made, "written")
    return StateRecord(
        delta_E=h5.read_array(group, "delta_E"),
        delta_U=h5.read_array(group, "delta_U"),
        W=h5.read_float(group, "W"),
        M_e=h5.read_float(group, "M_e"),
        X=h5.read_array(group, "X"),
        xi=h5.read_float(group, "xi"),
        j_e=h5.read_int(group, "j_e"),
        provenance=provenance,
        xi_form=none_if_nan(h5.read_float(group, "xi_form")),
        zones=zones_from_mappings(h5.read_mapping(group, "zones")["zones"]),
        E_whole=none_if_all_nan(h5.read_array(group, "E_whole")) if "E_whole" in h5.column_names(group) else None,
    )


def none_if_nan(value: float) -> float | None:
    """A float attribute that stands for `None` when NaN."""
    return None if math.isnan(value) else value


def none_if_all_nan(values: FloatArray) -> FloatArray | None:
    """An array that stands for `None` when every entry is NaN: `E_whole` with no cell stored whole."""
    return None if bool(np.all(np.isnan(values))) else values


def zones_as_mappings(zones: tuple[Zone, ...]) -> list[dict[str, float]]:
    """The zones as plain mappings, for a JSON attribute or a snapshot column."""
    return [dataclasses.asdict(zone) for zone in zones]


def zones_from_mappings(mappings: list[dict[str, float]]) -> tuple[Zone, ...]:
    """The zones back from their mappings."""
    return tuple(Zone(**mapping) for mapping in mappings)


def write_initial(path: Path, record: StateRecord) -> None:
    """Write the initial-data file from a record; `StateRecord.of` makes one from a state, its faces and its time."""
    with h5.create_file(path) as f:
        write_record(f, record)


def read_initial(path: Path) -> StateRecord:
    """Read an initial-data file."""
    with h5.open_file(path) as f:
        return read_record(f)

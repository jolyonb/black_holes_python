"""Tests of pbh.records and the h5 shim: the state record round-trips, in the initial file and in any group."""

import re
from pathlib import Path

import h5py
import numpy as np
import pytest

from pbh import h5
from pbh.eos import RADIATION, Background, EquationOfState
from pbh.geometry import Geometry
from pbh.initial import GrowingMode
from pbh.maps import IdentityMap, Map, SinhStretch, Zone
from pbh.profiles import Gaussian
from pbh.records import (
    FORMAT,
    VERSION,
    StateRecord,
    read_initial,
    read_record,
    shell_volumes,
    write_initial,
    write_record,
)
from pbh.state import State

RAD = EquationOfState(RADIATION)


def gaussian_datum(m: Map, N: int = 60) -> tuple[State, Geometry]:
    geo = Geometry.of(*m.radii(0.0, N))
    mode, _ = GrowingMode.from_profile(Gaussian(A=0.2, ell=2.0), "m", 12.0, Background.at(RAD, 0.0))
    return mode.state(geo, Background.at(RAD, 0.0)), geo


@pytest.mark.parametrize("m", [IdentityMap(12.0), SinhStretch(12.0, scale=3.0)])
def test_the_initial_file_round_trips_the_state_the_grid_the_time_and_the_provenance(tmp_path: Path, m: Map):
    state, geo = gaussian_datum(m)
    path = tmp_path / "run.initial.h5"
    provenance = {"method": "growing_mode", "profile": "gaussian", "A": 0.2, "ell": 2.0, "nested": {"seed": 7}}
    X = geo.X[: geo.N + 1]
    write_initial(path, StateRecord.of(state, X, 0.0, 0, provenance))
    record = read_initial(path)
    assert np.array_equal(record.delta_E, state.E - geo.dV)  # the deviation, stored as formed
    assert np.array_equal(record.delta_U, state.U - X)
    assert record.state.E == pytest.approx(state.E, rel=1e-15)
    assert record.state.U == pytest.approx(state.U, rel=1e-15)
    assert np.array_equal(record.deviation.E, record.delta_E)
    assert record.state.W == state.W
    assert record.state.M_e == 0.0
    assert np.array_equal(shell_volumes(X), geo.dV)
    assert np.array_equal(record.X, geo.X[: geo.N + 1])
    assert record.xi == 0.0
    assert record.j_e == 0
    assert {k: record.provenance[k] for k in provenance} == provenance
    assert re.fullmatch(r"[0-9a-f]{12}(-dirty)?|unknown", record.provenance["code_commit"])
    assert record.provenance["written"].endswith("+00:00")


def test_a_record_in_any_group_is_the_same_record_so_checkpoints_share_it(tmp_path: Path):
    state, geo = gaussian_datum(IdentityMap(12.0))
    excised = State(E=state.E, U=state.U, W=state.W, M_e=0.37)
    path = tmp_path / "run.evolution.h5"
    with h5py.File(path, "w") as f:
        record = StateRecord.of(excised, geo.X[: geo.N + 1], 3.5, 4, {"step": 12})
        write_record(h5.create_group(f, "checkpoints/000012"), record)
    with h5py.File(path, "r") as f:
        group = f["checkpoints/000012"]
        assert isinstance(group, h5py.Group)
        record = read_record(group)
    assert record.state.M_e == 0.37
    assert record.xi == 3.5
    assert record.j_e == 4
    assert record.provenance["step"] == 12


def test_the_provenance_accepts_numpy_values_and_refuses_what_json_cannot_hold(tmp_path: Path):
    state, geo = gaussian_datum(IdentityMap(12.0))
    path = tmp_path / "run.initial.h5"
    # np.float64 is a float to json already; np.int64 is not, and goes through the conversion
    provenance = {"ratio": np.float64(0.036), "N": np.int64(60), "fractions": np.array([1e-9, 2e-8])}
    write_initial(path, StateRecord.of(state, geo.X[: geo.N + 1], 0.0, 0, provenance))
    record = read_initial(path)
    assert record.provenance["ratio"] == 0.036
    assert record.provenance["N"] == 60
    assert record.provenance["fractions"] == [1e-9, 2e-8]
    with pytest.raises(TypeError, match="not JSON-serialisable"):
        write_initial(tmp_path / "bad.initial.h5", StateRecord.of(state, geo.X[: geo.N + 1], 0.0, 0, {"geo": geo}))


def test_a_state_that_does_not_fit_the_grid_and_a_foreign_file_are_refused(tmp_path: Path):
    state, _ = gaussian_datum(IdentityMap(12.0))
    other = Geometry.of(*IdentityMap(12.0).radii(0.0, 30))
    misfit = StateRecord(state.E - 1.0, state.U, 0.0, 0.0, other.X[:31], 0.0, 0, {})
    with pytest.raises(ValueError, match="does not fit the grid"):
        write_initial(tmp_path / "x.h5", misfit)
    foreign = tmp_path / "foreign.h5"
    with h5py.File(foreign, "w") as f:
        h5.write_text(f, "format", "something else")
        h5.write_int(f, "version", VERSION)
    with pytest.raises(ValueError, match="not a pbh state record"):
        read_initial(foreign)
    assert FORMAT == "pbh-state"


def test_the_formation_time_and_the_zones_round_trip(tmp_path: Path):
    state, geo = gaussian_datum(IdentityMap(12.0))
    zones = (Zone(xi_on=4.7, tau_on=0.3, x_t=0.2, Delta_t=0.075), Zone(xi_on=6.0, tau_on=0.3, x_t=0.5, Delta_t=0.15))
    record = StateRecord.of(state, geo.X[: geo.N + 1], 6.5, 0, {}, xi_form=4.6, zones=zones)
    path = tmp_path / "zoned.initial.h5"
    write_initial(path, record)
    again = read_initial(path)
    assert again.xi_form == 4.6
    assert again.zones == zones
    plain = read_initial(tmp_path / "plain.initial.h5") if False else None
    assert plain is None
    write_initial(tmp_path / "plain.initial.h5", StateRecord.of(state, geo.X[: geo.N + 1], 0.0, 0, {}))
    plain = read_initial(tmp_path / "plain.initial.h5")
    assert plain.xi_form is None
    assert plain.zones == ()

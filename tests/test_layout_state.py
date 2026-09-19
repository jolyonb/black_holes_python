"""Tests of pbh.layout and pbh.state: the packed order, the retained ranges, and the FRW state and rate."""

import numpy as np
import pytest

from pbh.geometry import Geometry
from pbh.layout import Layout
from pbh.maps import IdentityMap, PinnedMap, SinhStretch, face_labels
from pbh.state import State, frw_rate, frw_state, is_finite

N = 6


def random_state(rng: np.random.Generator) -> State:
    U = rng.normal(size=N + 1)
    U[0] = 0.0
    return State(E=rng.uniform(0.5, 1.5, size=N), U=U, W=float(rng.normal()), M_e=float(rng.uniform(1.0, 2.0)))


# --- the layout ---


def test_before_excision_the_retained_ranges_and_size_are_as_drawn():
    lay = Layout(N)
    assert not lay.excised
    assert lay.cells == slice(0, N)
    assert lay.faces == slice(0, N + 1)
    assert lay.faces_evolved == slice(1, N + 1)  # U_0 = 0 is never evolved
    assert lay.size == N + N + 1  # N energies, N face velocities, W


def test_after_excision_the_retained_ranges_and_size_are_as_drawn():
    lay = Layout(N, j_e=2)
    assert lay.excised
    assert lay.cells == slice(2, N)
    assert lay.faces == slice(2, N + 1)
    assert lay.faces_evolved == slice(2, N + 1)  # the excision face is an ordinary evolved face
    assert lay.size == (N - 2) + (N + 1 - 2) + 1 + 1  # energies, velocities, M_e, W


def test_too_few_cells_are_refused():
    with pytest.raises(ValueError, match="two cells"):
        Layout(1)


@pytest.mark.parametrize("j_e", [-1, N, N + 3])
def test_an_excision_face_outside_the_grid_is_refused(j_e: int):
    with pytest.raises(ValueError, match="0 <= j_e < N"):
        Layout(N, j_e)


def test_pack_puts_the_unknowns_in_the_documented_order():
    rng = np.random.default_rng(1)
    s = random_state(rng)
    y = Layout(N).pack(s)
    assert np.array_equal(y, np.concatenate((s.E, s.U[1:], [s.W])))
    y_e = Layout(N, j_e=2).pack(s)
    assert np.array_equal(y_e, np.concatenate((s.E[2:], s.U[2:], [s.M_e, s.W])))


@pytest.mark.parametrize("j_e", [0, 1, 3])
def test_pack_then_unpack_returns_the_retained_entries_and_nan_elsewhere(j_e: int):
    rng = np.random.default_rng(2)
    s = random_state(rng)
    lay = Layout(N, j_e)
    back = lay.unpack(lay.pack(s))
    assert np.array_equal(back.E[j_e:], s.E[j_e:])
    assert np.array_equal(back.U[max(j_e, 1) :], s.U[max(j_e, 1) :])
    assert back.W == s.W
    assert back.M_e == (s.M_e if j_e > 0 else 0.0)
    assert np.all(np.isnan(back.E[:j_e]))
    if j_e == 0:
        assert back.U[0] == 0.0
    else:
        assert np.all(np.isnan(back.U[:j_e]))


def test_unpack_then_pack_is_the_identity_on_vectors():
    rng = np.random.default_rng(3)
    for lay in (Layout(N), Layout(N, j_e=2)):
        y = rng.normal(size=lay.size)
        assert np.array_equal(lay.pack(lay.unpack(y)), y)


def test_a_vector_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="length"):
        Layout(N).unpack(np.zeros(3))


# --- the state ---


def test_mismatched_cell_and_face_counts_are_refused():
    with pytest.raises(ValueError, match="N \\+ 1"):
        State(E=np.zeros(N), U=np.zeros(N), W=0.0)


def test_the_state_is_frozen():
    s = random_state(np.random.default_rng(4))
    with pytest.raises(AttributeError):
        s.W = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("j_e", [0, 2])
def test_is_finite_looks_only_at_the_retained_entries(j_e: int):
    s = Layout(N, j_e).unpack(np.ones(Layout(N, j_e).size))  # NaN below j_e by construction
    assert is_finite(s, j_e)
    E_bad = s.E.copy()
    E_bad[N - 1] = np.inf
    assert not is_finite(State(E=E_bad, U=s.U, W=s.W, M_e=s.M_e), j_e)
    assert not is_finite(State(E=s.E, U=s.U, W=np.nan, M_e=s.M_e), j_e)


# --- the FRW state and rate ---


@pytest.mark.parametrize("m", [IdentityMap(), SinhStretch(scale=2.0)])
def test_the_frw_state_on_a_static_map_has_the_printed_values_and_no_rate(m: IdentityMap | SinhStretch):
    geo = Geometry.of(*m.at(0.0, face_labels(N, 3.0)))
    s = frw_state(geo, j_e=2)
    assert np.array_equal(s.E, geo.dV)
    assert np.array_equal(s.U, geo.X)
    assert s.W == 0.0
    assert s.M_e == geo.X[2] ** 3
    r = frw_rate(geo, j_e=2)
    assert np.all(r.E == 0.0)
    assert np.all(r.U == 0.0)
    assert r.W == 0.0
    assert r.M_e == 0.0


def test_the_frw_rate_on_the_pinned_map_is_the_exact_derivative():
    alpha = 0.5
    m = PinnedMap(SinhStretch(scale=2.0), alpha=alpha)
    x = face_labels(N, 3.0)
    geo = Geometry.of(*m.at(0.7, x))
    r = frw_rate(geo, j_e=2)
    # Every radius scales as e^(-alpha xi): dV as e^(-3 alpha xi), X as e^(-alpha xi), X_e^3 as e^(-3 alpha xi).
    assert r.E == pytest.approx(-3.0 * alpha * geo.dV, rel=1e-14)
    assert r.U == pytest.approx(-alpha * geo.X, rel=1e-14)
    assert r.M_e == pytest.approx(-3.0 * alpha * geo.X[2] ** 3, rel=1e-14)
    # And it is the central difference of the FRW state in time.
    eps = 1e-6
    later = frw_state(Geometry.of(*m.at(0.7 + eps, x)), j_e=2)
    earlier = frw_state(Geometry.of(*m.at(0.7 - eps, x)), j_e=2)
    assert r.E == pytest.approx((later.E - earlier.E) / (2 * eps), rel=1e-8)
    assert r.M_e == pytest.approx((later.M_e - earlier.M_e) / (2 * eps), rel=1e-8)


def test_the_deviation_of_frw_from_itself_is_the_zero_vector():
    geo = Geometry.of(*SinhStretch(scale=2.0).at(0.0, face_labels(N, 3.0)))
    lay = Layout(N, j_e=1)
    assert np.all(lay.pack(frw_state(geo, lay.j_e)) - lay.pack(frw_state(geo, lay.j_e)) == 0.0)

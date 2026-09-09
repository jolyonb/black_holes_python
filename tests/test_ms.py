"""Tests for Misner-Sharp evolution."""

import io
from pathlib import Path

import numpy as np
import pytest

from pbh.base import EvolverError, FloatArray, Status
from pbh.initial import compute_deltam0, growingmode, makegrid
from pbh.ms import MS, MSCommon, MSEulerian, MSLagrangian

InitialData = tuple[FloatArray, FloatArray, FloatArray]

HANDLERS = [MSEulerian, MSLagrangian]


def test_haystack_conserves_total_away_from_edges() -> None:
    signal = np.zeros(41)
    signal[20] = 1.0
    smoothed = MSCommon.haystack(signal)
    assert smoothed.sum() == pytest.approx(1.0)
    assert smoothed[20] == pytest.approx(8 / 48)
    assert smoothed[15] == pytest.approx(1 / 48)
    assert smoothed[14] == 0
    # Symmetric
    assert smoothed[15:20] == pytest.approx(smoothed[21:26][::-1])


def test_evolver_requires_initial_conditions() -> None:
    driver = MS(eomhandler=MSEulerian)
    assert driver.status == Status.NEEDS_INITIALIZING
    with pytest.raises(ValueError, match="set_initial_conditions"):
        _ = driver.integrator
    with pytest.raises(ValueError, match="status"):
        driver.evolve(1.0)


def test_mismatched_field_lengths_rejected() -> None:
    driver = MS(eomhandler=MSEulerian)
    with pytest.raises(ValueError, match="same number"):
        driver.set_initial_conditions(0.0, np.ones(5), np.ones(5), np.ones(4))


@pytest.mark.parametrize("handler", HANDLERS)
def test_initial_state_quantities(handler: type[MSCommon], small_initial_data: InitialData) -> None:
    r, u, m = small_initial_data
    driver = MS(eomhandler=handler, viscosity=2)
    driver.set_initial_conditions(0.0, r, u, m)
    assert driver.status == Status.READY
    eom = driver.eomhandler
    assert eom.r == pytest.approx(r)
    assert eom.u == pytest.approx(u)
    assert eom.m == pytest.approx(m)
    assert eom.xi == 0.0
    assert eom.a == 1.0
    assert eom.H == 1.0
    assert np.all(eom.rho > 0)
    assert np.all(eom.gamma2 > 0)
    # No shock initially, so no artificial viscosity
    assert not eom.viscosity_present
    assert np.all(eom.Q == 0)
    assert eom.ephi == pytest.approx(eom.rho ** (-1 / 4))
    assert driver.timeouttime == pytest.approx(0.8278 + 2 * np.log(r[-1]))


@pytest.mark.parametrize("handler", HANDLERS)
def test_cache_is_invalidated_when_fields_change(handler: type[MSCommon], small_initial_data: InitialData) -> None:
    r, u, m = small_initial_data
    driver = MS(eomhandler=handler)
    driver.set_initial_conditions(0.0, r, u, m)
    eom = driver.eomhandler
    rho_before = eom.rho
    assert eom.rho is rho_before  # cached
    eom.set_fields(0.0, driver.integrator.values * 1.01)
    assert eom.rho is not rho_before
    assert eom.m == pytest.approx(m * 1.01)


@pytest.mark.parametrize("handler", HANDLERS)
def test_short_evolution_is_stable(handler: type[MSCommon], small_initial_data: InitialData) -> None:
    r, u, m = small_initial_data
    driver = MS(eomhandler=handler, viscosity=2)
    driver.set_initial_conditions(0.0, r, u, m)
    out = io.StringIO()
    driver.drive(output_step=0.5, file_handle=out, max_time=1.0)
    assert driver.status == Status.TIMEOUT
    assert driver.xi == pytest.approx(1.0)
    # Background should still be background at the outer edge
    assert driver.eomhandler.m[-1] == pytest.approx(1.0, abs=1e-3)
    # Three output blocks (xi = 0, 0.5, 1.0), each with a header, a line per gridpoint, and a blank line
    blocks = out.getvalue().strip().split("\n\n")
    assert len(blocks) == 3
    assert all(len(block.splitlines()) == len(r) + 1 for block in blocks)
    assert blocks[0].splitlines()[0].startswith("# index\tr\tu\tm\trho\t")


@pytest.mark.parametrize("handler", HANDLERS)
def test_outer_boundary_condition_is_applied(handler: type[MSCommon], small_initial_data: InitialData) -> None:
    r, u, m = small_initial_data
    driver = MS(eomhandler=handler)
    driver.set_initial_conditions(0.0, r, u, m)
    eom = driver.eomhandler
    assert eom.udot_lagrangian[-1] == eom.udot_outer_boundary
    rdot, udot, mdot = eom.derivatives()
    assert rdot.shape == udot.shape == mdot.shape == r.shape


@pytest.mark.parametrize("handler", HANDLERS)
def test_background_is_static(handler: type[MSCommon]) -> None:
    """The unperturbed background (u = r, m = 1) is a fixed point of the rescaled equations, boundary included."""
    grid = makegrid(gridpoints=50, squeeze=0, Amax=10)
    r, u, m = growingmode(grid, np.zeros_like(grid))
    driver = MS(eomhandler=handler)
    driver.set_initial_conditions(0.0, r, u, m)
    eom = driver.eomhandler
    assert eom.udot_outer_boundary == pytest.approx(0, abs=1e-12)
    for dot in eom.derivatives():
        assert dot == pytest.approx(0, abs=1e-12)


def test_eulerian_and_lagrangian_agree_at_early_times(small_initial_data: InitialData) -> None:
    """Both schemes are discretisations of the same equations; the central mass function should agree closely."""
    r, u, m = small_initial_data
    results: dict[str, float] = {}
    for handler in HANDLERS:
        driver = MS(eomhandler=handler)
        driver.set_initial_conditions(0.0, r, u, m)
        driver.drive(output_step=1.0, file_handle=io.StringIO(), max_time=1.0)
        results[handler.__name__] = float(driver.eomhandler.m[0])
    assert results["MSEulerian"] == pytest.approx(results["MSLagrangian"], rel=1e-3)


@pytest.mark.parametrize(
    ("handler", "expected_xi"),
    [(MSEulerian, 4.321432333547402), (MSLagrangian, 4.321234132034417)],
)
@pytest.mark.slow
def test_black_hole_formation_golden(handler: type[MSCommon], expected_xi: float) -> None:
    """Golden regression test: horizon formation time for a supercritical perturbation (recorded 2026-09-08)."""
    grid = makegrid(gridpoints=300, squeeze=2, Amax=10)
    r, u, m = growingmode(grid, compute_deltam0(grid, amplitude=0.19))
    driver = MS(eomhandler=handler, black_hole_check=True, viscosity=2)
    driver.set_initial_conditions(0.0, r, u, m)
    driver.drive(output_step=0.5, file_handle=io.StringIO(), max_time=6)
    assert driver.status == Status.BLACKHOLE_FORMED
    assert driver.xi == pytest.approx(expected_xi, rel=1e-9)
    eom = driver.eomhandler
    assert np.any((eom.horizon >= 1) & (eom.u < 0))


@pytest.mark.slow
def test_viscosity_triggers_in_supercritical_run() -> None:
    grid = makegrid(gridpoints=300, squeeze=2, Amax=10)
    r, u, m = growingmode(grid, compute_deltam0(grid, amplitude=0.19))
    driver = MS(eomhandler=MSEulerian, black_hole_check=True, viscosity=2)
    driver.set_initial_conditions(0.0, r, u, m)
    driver.drive(output_step=0.5, file_handle=io.StringIO(), max_time=6)
    eom = driver.eomhandler
    assert eom.viscosity_present
    assert np.any(eom.Q > 0)
    # The envelope switches Q off near the outer boundary
    assert eom.Q[-1] == 0
    np.testing.assert_allclose(eom.P, eom.rho / 3 + eom.rho * eom.Q)


@pytest.mark.slow
def test_viscosity_disabled_when_none() -> None:
    grid = makegrid(gridpoints=300, squeeze=2, Amax=10)
    r, u, m = growingmode(grid, compute_deltam0(grid, amplitude=0.19))
    driver = MS(eomhandler=MSEulerian, viscosity=None)
    driver.set_initial_conditions(0.0, r, u, m)
    driver.drive(output_step=1.0, file_handle=io.StringIO(), max_time=3)
    eom = driver.eomhandler
    assert not eom.viscosity_present
    assert np.all(eom.Q == 0)


@pytest.mark.slow
def test_enforce_timeout_stops_at_timeout_time() -> None:
    # A subcritical perturbation that will not form a black hole
    grid = makegrid(gridpoints=150, squeeze=2, Amax=10)
    r, u, m = growingmode(grid, compute_deltam0(grid, amplitude=0.1))
    driver = MS(eomhandler=MSEulerian, enforce_timeout=True)
    driver.set_initial_conditions(0.0, r, u, m)
    driver.drive(output_step=0.5, file_handle=io.StringIO(), max_time=20)
    assert driver.status == Status.TIMEOUT
    assert driver.timeouttime < driver.xi <= driver.timeouttime + 0.5


def test_negative_gamma2_sets_status() -> None:
    grid = makegrid(gridpoints=100, squeeze=2, Amax=10)
    r, u, m = growingmode(grid, compute_deltam0(grid, amplitude=0.25))
    driver = MS(eomhandler=MSEulerian)
    driver.set_initial_conditions(0.0, r, u, m)
    with pytest.raises(EvolverError) as excinfo:
        _ = driver.eomhandler.gamma2
    assert excinfo.value.status == Status.NEGATIVE_GAMMA2
    # The evolver only learns about the error when it catches it during evolution
    assert driver.status == Status.READY
    assert driver.evolve(1.0) is True
    assert driver.status == Status.NEGATIVE_GAMMA2


def test_unphysical_initial_data_stops_drive_gracefully(tmp_path: Path) -> None:
    """An EvolverError raised while writing output must be caught, not escape from drive()."""
    grid = makegrid(gridpoints=100, squeeze=2, Amax=10)
    r, u, m = growingmode(grid, compute_deltam0(grid, amplitude=0.25))
    driver = MS(eomhandler=MSEulerian)
    driver.set_initial_conditions(0.0, r, u, m)
    datafile = tmp_path / "out.dat"
    with datafile.open("w") as f:
        driver.drive(output_step=0.5, file_handle=f, max_time=1.0)
    assert driver.status == Status.NEGATIVE_GAMMA2
    assert driver.msg == "NEGATIVE_GAMMA2"
    assert driver.xi == 0.0


def test_load_initial_conditions_round_trip(tmp_path: Path, small_initial_data: InitialData) -> None:
    r, u, m = small_initial_data
    driver = MS(eomhandler=MSEulerian)
    driver.set_initial_conditions(0.0, r, u, m)
    datafile = tmp_path / "out.dat"
    with datafile.open("w") as f:
        driver.drive(output_step=0.5, file_handle=f, max_time=0.5)
    # Evolve a little further so the first block differs from the final state
    assert driver.xi == pytest.approx(0.5)

    loaded = MS(eomhandler=MSEulerian)
    loaded.load_initial_conditions(str(datafile))
    assert loaded.status == Status.READY
    assert loaded.xi == 0.0
    assert loaded.eomhandler.r == pytest.approx(r)
    assert loaded.eomhandler.u == pytest.approx(u)
    assert loaded.eomhandler.m == pytest.approx(m)

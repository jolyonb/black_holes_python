"""Tests for Misner-Sharp evolution."""

import io
import math
from abc import ABC
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy.special import spherical_jn

from pbh.base import EOSParameter, EvolverError, FloatArray, Status, alpha_of_w, as_rational_w, cached_property
from pbh.initial import compute_deltam0, growingmode, makegrid
from pbh.ms import MS, MSCommon, MSEulerian, MSLagrangian
from pbh.output import GnuplotWriter

InitialData = tuple[FloatArray, FloatArray, FloatArray]

HANDLERS = [MSEulerian, MSLagrangian]
#: Equations of state to vary: the general-w tests must hold for all of these, and only 1/3 has the exact boundary
W_VALUES: list[EOSParameter] = [0.2, Fraction(1, 3), 0.6]


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
    driver.drive(output_step=0.5, writer=GnuplotWriter(out), max_time=1.0)
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


def background(gridpoints: int = 50, Amax: float = 10) -> InitialData:
    """The unperturbed FRW background (r = grid, u = r, m = 1) on a uniform grid."""
    grid = makegrid(gridpoints=gridpoints, squeeze=0, Amax=Amax)
    return grid, grid.copy(), np.ones_like(grid)


@pytest.mark.parametrize("w", W_VALUES)
@pytest.mark.parametrize("handler", HANDLERS)
def test_background_is_static(handler: type[MSCommon], w: EOSParameter) -> None:
    """The unperturbed background (u = r, m = 1) is a fixed point of the rescaled equations for every w.

    Several w-dependent terms must cancel for this to hold (in udot, (1 - alpha) r = alpha (r/2)(1 + 3w) since
    2(1 - alpha) = alpha(1 + 3w)), so a dropped factor of alpha or w shows up here. The outer boundary condition only
    exists for w = 1/3, so for other w the interior equation is checked at every row and the boundary must refuse.
    """
    r, u, m = background()
    driver = MS(eomhandler=handler, w=w)
    driver.set_initial_conditions(0.0, r, u, m)
    eom = driver.eomhandler
    assert eom.rho == pytest.approx(1.0)
    np.testing.assert_allclose(eom.P, eom.w)
    assert eom.ephi == pytest.approx(1.0)
    assert eom.rdot_lagrangian == pytest.approx(0, abs=1e-12)
    assert eom.mdot_lagrangian == pytest.approx(0, abs=1e-12)
    assert eom.udot_interior == pytest.approx(0, abs=1e-12)
    if eom.is_radiation:
        assert eom.udot_outer_boundary == pytest.approx(0, abs=1e-12)
        for dot in eom.derivatives():
            assert dot == pytest.approx(0, abs=1e-12)
    else:
        with pytest.raises(NotImplementedError, match="w = 1/3"):
            eom.derivatives()


@pytest.mark.parametrize("w", W_VALUES)
@pytest.mark.parametrize("handler", HANDLERS)
def test_background_scalings_for_general_w(handler: type[MSCommon], w: EOSParameter) -> None:
    """a = e^(alpha xi) and H = e^(-xi) for every w; the (H a R_H)^2 sites must not use the alpha = 1/2 coincidence."""
    r, u, m = background()
    xi = 1.0
    driver = MS(eomhandler=handler, w=w)
    driver.set_initial_conditions(xi, r, u, m)
    eom = driver.eomhandler
    alpha = eom.alpha
    assert alpha == float(alpha_of_w(as_rational_w(w)))
    assert eom.a == pytest.approx(np.exp(alpha * xi), rel=1e-14)
    np.testing.assert_allclose(eom.H, np.exp(-xi), rtol=1e-14)
    assert eom.Ha2 == pytest.approx(np.exp(2 * (alpha - 1) * xi), rel=1e-14)
    assert eom.horizon == pytest.approx(r * r * np.exp(2 * (alpha - 1) * xi), rel=1e-14)
    # For the background, gamma^2 = e^(2(1-alpha)xi) + r^2 - r^2
    assert eom.gamma2 == pytest.approx(np.exp(2 * (1 - alpha) * xi), rel=1e-12)
    # Sound speed alpha sqrt(w) e^phi gamma, e^phi = 1 on the background
    assert eom.c_characteristic == pytest.approx(alpha * np.sqrt(eom.w) * np.exp((1 - alpha) * xi), rel=1e-12)
    # The background is static at any xi, not just xi = 0
    assert eom.mdot_lagrangian == pytest.approx(0, abs=1e-12)
    assert eom.udot_interior == pytest.approx(0, abs=1e-12)


class FreeOuterBoundary(MSCommon, ABC):
    """Test-only outer boundary: the interior equation with drhodr = 0 (a zero pressure gradient condition).

    The exact outgoing-wave condition exists only for w = 1/3. This crude replacement is exact for FRW, and on a
    large domain its error cannot reach the inner half within Delta xi = 1 (sound travels less than 0.5 in r).
    """

    @cached_property
    def udot_outer_boundary(self) -> float:
        return float(self.udot_interior[-1])


class FreeEulerian(FreeOuterBoundary, MSEulerian):
    pass


class FreeLagrangian(FreeOuterBoundary, MSLagrangian):
    pass


@pytest.mark.parametrize("w", W_VALUES)
@pytest.mark.parametrize("handler", [FreeEulerian, FreeLagrangian])
def test_superhorizon_growth_exponent(handler: type[MSCommon], w: EOSParameter) -> None:
    """A superhorizon perturbation grows as e^(2(1-alpha) xi): e^xi for radiation, faster for stiffer fluids.

    The exponent is (2 + 6w)/(3(1 + w)): 8/9 at w = 0.2, 1 at w = 1/3, 7/6 at w = 0.6. A long wavelength
    (Amax = 40, k = pi/Amax) keeps the (k c_s tau)^2 corrections below 1e-3; the initial data is the first-order
    growing mode dU = -(alpha/2) dm of the fundamental Bessel mode (drho = eps j0(k r)); only the inner half of the
    domain is checked since the outer boundary condition is not the exact one.
    """
    alpha = float(alpha_of_w(as_rational_w(w)))
    Amax, n, eps = 40.0, 200, 1e-5
    r = makegrid(gridpoints=n, squeeze=0, Amax=Amax)
    k = np.pi / Amax
    dm0 = eps * 3 * spherical_jn(1, k * r) / (k * r)  # dm(r -> 0) = eps
    driver = MS(eomhandler=handler, black_hole_check=False, viscosity=None, w=w)
    driver.set_initial_conditions(0.0, r.copy(), r * (1 - alpha / 2 * dm0), 1 + dm0)
    driver.drive(output_step=1.0, writer=None, max_time=1.0)
    assert driver.xi == pytest.approx(1.0)
    eom = driver.eomhandler
    inner = r < 0.5 * Amax
    growth = np.log((eom.m[inner] - 1) / dm0[inner])
    assert growth[0] == pytest.approx(2 * (1 - alpha), abs=5e-3)
    # A single mode keeps its profile, so the whole inner region grows by the same factor
    assert growth == pytest.approx(2 * (1 - alpha), abs=5e-3)


def test_non_radiation_guards() -> None:
    """The theory that exists only for w = 1/3 refuses other w rather than silently using radiation numbers."""
    r, u, m = background()
    with pytest.raises(NotImplementedError, match="enforce_timeout"):
        MS(eomhandler=MSEulerian, enforce_timeout=True, w=0.2)
    driver = MS(eomhandler=MSEulerian, w=0.2)
    driver.set_initial_conditions(0.0, r, u, m)
    assert driver.timeouttime == math.inf
    eom = driver.eomhandler
    with pytest.raises(NotImplementedError, match="outgoing-wave"):
        _ = eom.udot_outer_boundary
    with pytest.raises(NotImplementedError, match="w = 1/3"):
        _ = eom.udot_lagrangian
    # The interior equation is still available
    assert eom.udot_interior.shape == r.shape


@pytest.mark.parametrize("w", [Fraction(1, 3), 1 / 3, "1/3", "2/6", 0.3333333333])
def test_radiation_constants_are_exact(w: EOSParameter) -> None:
    """Every spelling of w = 1/3 yields the same exact constants, so the radiation numerics are bitwise unchanged."""
    eom = MSEulerian(w=w)
    assert eom.w_exact == Fraction(1, 3)
    assert eom.is_radiation
    assert eom.w == 1 / 3
    assert eom.alpha == 0.5
    assert eom.inv_alpha == 2.0
    assert eom.inv_w == 3.0
    assert eom.lapse_exponent == 0.25  # e^phi = rho^(-1/4)
    assert eom.inv_cs_factor == np.sqrt(12)  # the old hard-coded divisor in c_characteristic
    # Computed once in __init__ and stored as plain floats, not recomputed on every access
    for name in ("w", "inv_w", "alpha", "inv_alpha", "lapse_exponent", "inv_cs_factor"):
        assert isinstance(vars(eom)[name], float)


def test_eos_constants_for_general_w() -> None:
    assert alpha_of_w(Fraction(1, 3)) == Fraction(1, 2)  # radiation
    assert alpha_of_w(Fraction(0)) == Fraction(2, 3)  # dust
    assert alpha_of_w(Fraction(1)) == Fraction(1, 3)  # stiff
    assert as_rational_w("0.2") == as_rational_w(0.2) == Fraction(1, 5)
    eom = MSLagrangian(w=0.2)
    assert eom.w_exact == Fraction(1, 5)
    assert not eom.is_radiation
    assert eom.alpha == float(Fraction(5, 9))
    assert eom.inv_alpha == float(Fraction(9, 5))
    assert eom.inv_w == 5.0
    assert eom.lapse_exponent == float(Fraction(1, 6))
    assert eom.inv_cs_factor == pytest.approx(1 / (eom.alpha * np.sqrt(0.2)), rel=1e-15)
    for bad in (0, -0.5, 1.5, "4/3"):
        with pytest.raises(ValueError, match="0 < w <= 1"):
            as_rational_w(bad)
    # A positive float below the canonicalisation resolution rounds to 0; the message says so
    with pytest.raises(ValueError, match=r"got 1e-07 \(a float is rounded .* 10\*\*6: 0\)"):
        as_rational_w(1e-7)
    assert as_rational_w(Fraction(1, 10_000_000)) == Fraction(1, 10_000_000)  # exact rationals are not rounded
    with pytest.raises(ValueError, match="0 < w <= 1"):
        MSEulerian(w=0)


class ForcedViscosity(MSEulerian):
    """Marks artificial viscosity as present with Q = 0, exercising the viscous branches of ephi, P and dPdr."""

    def _computeQ(self) -> None:
        self._cacheQ(True, np.zeros_like(self.r), np.zeros_like(self.r))


@pytest.mark.parametrize("w", W_VALUES)
def test_viscous_lapse_split_cancels_when_q_vanishes(w: EOSParameter) -> None:
    """With Q = 0 the viscous e^phi must reduce to the analytic rho^(-w/(1+w)) at machine precision for every w."""
    grid = makegrid(gridpoints=150, squeeze=2, Amax=10)
    m = 1 + 0.1 * np.exp(-grid * grid / 8)
    driver = MS(eomhandler=ForcedViscosity, viscosity=2, w=w)
    driver.set_initial_conditions(0.0, grid, grid.copy(), m)
    eom = driver.eomhandler
    assert eom.viscosity_present
    assert np.array_equal(eom.ephi, eom.rho ** (-eom.lapse_exponent))
    np.testing.assert_allclose(eom.P, eom.w * eom.rho, rtol=1e-15)
    assert eom.dPdr == pytest.approx(eom.w * eom.drhodr, rel=1e-15)


def test_eulerian_and_lagrangian_agree_at_early_times(small_initial_data: InitialData) -> None:
    """Both schemes are discretisations of the same equations; the central mass function should agree closely."""
    r, u, m = small_initial_data
    results: dict[str, float] = {}
    for handler in HANDLERS:
        driver = MS(eomhandler=handler)
        driver.set_initial_conditions(0.0, r, u, m)
        driver.drive(output_step=1.0, max_time=1.0)
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
    driver.drive(output_step=0.5, max_time=6)
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
    driver.drive(output_step=0.5, max_time=6)
    eom = driver.eomhandler
    assert eom.viscosity_present
    assert np.any(eom.Q > 0)
    # The envelope switches Q off near the outer boundary
    assert eom.Q[-1] == 0
    np.testing.assert_allclose(eom.P, eom.rho / 3 + eom.rho * eom.Q)


@pytest.mark.slow
@pytest.mark.parametrize("handler", HANDLERS)
def test_viscosity_envelope_is_resolution_independent(handler: type[MSCommon]) -> None:
    """The envelope is a fixed function of r, not of grid index."""
    envelopes: dict[int, tuple[FloatArray, FloatArray]] = {}
    for n in (300, 600):
        r = makegrid(gridpoints=n, squeeze=2, Amax=10)
        driver = MS(eomhandler=handler, viscosity=2)
        driver.set_initial_conditions(0.0, r, r.copy(), np.ones_like(r))
        envelopes[n] = (r, driver.eomhandler.Qenvelope)
    r3, e3 = envelopes[300]
    r6, e6 = envelopes[600]
    # Same function of the distance from the outer edge (the two grids' last points differ by half a cell)
    depth3, depth6 = r3[-1] - r3, r6[-1] - r6
    assert np.interp(depth3, depth6[::-1], e6[::-1]) == pytest.approx(e3, abs=1e-3)
    assert e3[-1] < 1e-4
    assert e6[-1] < 1e-4  # off at the boundary
    assert np.interp(r6[-1] - 1.0, r6, e6) == pytest.approx(0.5, abs=0.02)  # midpoint at the buffer distance
    assert e6[r6 < r6[-1] - 2.0].min() > 0.999  # fully on well inside


def test_viscosity_envelope_options() -> None:
    r = makegrid(gridpoints=300, squeeze=2, Amax=10)
    driver = MS(eomhandler=MSEulerian, viscosity=2, viscosity_buffer=3.0, viscosity_buffer_width=0.5)
    driver.set_initial_conditions(0.0, r, r.copy(), np.ones_like(r))
    envelope = driver.eomhandler.Qenvelope
    assert np.interp(r[-1] - 3.0, r, envelope) == pytest.approx(0.5, abs=0.02)
    assert np.interp(r[-1] - 3.0 + 0.5, r, envelope) == pytest.approx(1 / (np.e + 1), abs=0.02)


def test_viscosity_disabled_when_none() -> None:
    grid = makegrid(gridpoints=300, squeeze=2, Amax=10)
    r, u, m = growingmode(grid, compute_deltam0(grid, amplitude=0.19))
    driver = MS(eomhandler=MSEulerian, viscosity=None)
    driver.set_initial_conditions(0.0, r, u, m)
    driver.drive(output_step=1.0, max_time=3)
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
    driver.drive(output_step=0.5, max_time=20)
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
        driver.drive(output_step=0.5, writer=GnuplotWriter(f), max_time=1.0)
    assert driver.status == Status.NEGATIVE_GAMMA2
    assert driver.msg == "NEGATIVE_GAMMA2"
    assert driver.xi == 0.0


def test_load_initial_conditions_round_trip(tmp_path: Path, small_initial_data: InitialData) -> None:
    r, u, m = small_initial_data
    driver = MS(eomhandler=MSEulerian)
    driver.set_initial_conditions(0.0, r, u, m)
    datafile = tmp_path / "out.dat"
    with datafile.open("w") as f:
        driver.drive(output_step=0.5, writer=GnuplotWriter(f), max_time=0.5)
    # Evolve a little further so the first block differs from the final state
    assert driver.xi == pytest.approx(0.5)

    loaded = MS(eomhandler=MSEulerian)
    loaded.load_initial_conditions(str(datafile))
    assert loaded.status == Status.READY
    assert loaded.xi == 0.0
    assert loaded.eomhandler.r == pytest.approx(r)
    assert loaded.eomhandler.u == pytest.approx(u)
    assert loaded.eomhandler.m == pytest.approx(m)

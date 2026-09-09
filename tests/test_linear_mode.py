"""The code must reproduce the exact linear solution for a single spherical Bessel mode.

test_background_is_static pins the zeroth order; this pins first order. For w = 1/3 and tau = e^{xi/2}/sqrt(3), a
single mode of wavenumber k has the closed form

    drho = B k tau j0(k r) f1(k tau)
    dm   = B (3 tau / r) j1(k r) f1(k tau)
    dU   = -B (3 tau / (4 r)) j1(k r) (k tau f0(k tau) - 2 f1(k tau))

with (f0, f1) = (j0, j1) for the growing mode and (y0, y1) for the decaying mode (spherical Bessel functions).
For k tau << 1 the growing mode has dm ~ tau^2 ~ e^xi and dU ~ -dm/4, the familiar superhorizon growth; the
decaying mode falls like 1/tau. The initial areal-radius labelling is a gauge choice, so r = grid is valid initial
data for both handlers.
"""

from collections.abc import Callable

import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

from pbh.base import FloatArray
from pbh.initial import makegrid
from pbh.ms import MS, MSCommon, MSEulerian, MSLagrangian

Bessel = Callable[[FloatArray], FloatArray]

# (f1, x f0(x)) for each mode: k tau j0(k tau) = sin(k tau) and k tau y0(k tau) = -cos(k tau)
MODES: dict[str, tuple[Bessel, Bessel]] = {
    "growing": (lambda x: spherical_jn(1, x), np.sin),
    "decaying": (lambda x: spherical_yn(1, x), lambda x: -np.cos(x)),
}


def tau_of(xi: float) -> float:
    return float(np.exp(xi / 2) / np.sqrt(3))


def exact_dm(xi: float, r: FloatArray, k: float, B: float, f1: Bessel) -> FloatArray:
    tau = tau_of(xi)
    return B * 3 * tau / r * spherical_jn(1, k * r) * f1(np.asarray(k * tau))


def exact_dU(xi: float, r: FloatArray, k: float, B: float, f1: Bessel, xf0: Bessel) -> FloatArray:
    tau = tau_of(xi)
    x = np.asarray(k * tau)
    return -B * 3 * tau / (4 * r) * spherical_jn(1, k * r) * (xf0(x) - 2 * f1(x))


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("handler", [MSEulerian, MSLagrangian])
def test_single_bessel_mode_follows_linear_theory(handler: type[MSCommon], mode: str) -> None:
    f1, xf0 = MODES[mode]
    Amax, n, eps = 10.0, 200, 1e-5
    r = makegrid(gridpoints=n, squeeze=0, Amax=Amax)
    k = np.pi / Amax  # fundamental: drho(Amax) = 0
    t0 = tau_of(0.0)
    B = eps / (t0 * k * float(f1(np.asarray(k * t0))))  # normalise so dm(0, r -> 0) = eps
    m0 = 1 + exact_dm(0.0, r, k, B, f1)
    u0 = r * (1 + exact_dU(0.0, r, k, B, f1, xf0))

    driver = MS(eomhandler=handler, black_hole_check=False, enforce_timeout=False, viscosity=None)
    driver.set_initial_conditions(0.0, r.copy(), u0, m0)
    driver.drive(output_step=1.0, writer=None, max_time=1.0)
    assert driver.xi == pytest.approx(1.0)

    # The outgoing boundary condition lets the standing mode leak near the edge; sound travels about 0.37 in
    # Delta xi = 1, so the inner half of the domain is clean.
    eom = driver.eomhandler
    inner = r < 0.5 * Amax
    rr = eom.r[inner]  # Lagrangian: r has moved by O(eps), so use the current r
    dm_num = eom.m[inner] - 1
    dU_num = eom.u[inner] / rr - 1
    dm_ex = exact_dm(1.0, rr, k, B, f1)
    dU_ex = exact_dU(1.0, rr, k, B, f1, xf0)
    # Error budget: nonlinear corrections O(eps) = 1e-5; spatial truncation O((k h)^2) ~ 2e-4 at n = 200.
    assert dm_num == pytest.approx(dm_ex, rel=5e-4, abs=5e-4 * eps)
    assert dU_num == pytest.approx(dU_ex, rel=1e-3, abs=5e-4 * eps)

    if mode == "growing":
        # Headline number: the central growth factor is e^1 up to the (k tau)^2 / 10 correction, so 2.703 not 2.718
        growth = dm_num[0] / exact_dm(0.0, rr[:1], k, B, f1)[0]
        predicted = np.e * (1 - (k * tau_of(1.0)) ** 2 / 10) / (1 - (k * t0) ** 2 / 10)
        assert growth == pytest.approx(predicted, rel=5e-4)
    else:
        # The decaying mode must decay, not be projected onto the growing mode
        assert abs(dm_num[0]) < abs(exact_dm(0.0, rr[:1], k, B, f1)[0])

"""The exact linear modes of radiation (paper Section 5.2, eq:lin:mode, eq:lin:modem, eq:lin:modeU), for tests.

A mode with wavenumber `k` and amplitude `B` (the growing branch, `C = 0`) has, with `tau` the sound horizon and
`j_n` the spherical Bessel functions,

    delta_rho = tau k j_0(k R) B j_1(k tau),
    delta_m   = (3 tau / R) j_1(k R) B j_1(k tau),
    delta_U   = -(3 tau / 4 R) j_1(k R) B [j_1(k tau) - k tau j_2(k tau)],

and the cell contents of `1 + delta_rho` are exact in closed form because `int r^2 j_0(k r) dr = R^2 j_1(k R) / k`.
Choosing `k` so that `j_1(k X_N) = 0` makes `delta_U` vanish at the outer face for all time, so the held closure of
`outer.py` is exact there and the modes can be evolved without a boundary datum.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import spherical_jn

from pbh.eos import RADIATION, Background, EquationOfState
from pbh.geometry import Geometry
from pbh.kernels import KernelSettings
from pbh.layout import Layout
from pbh.maps import Map
from pbh.outer import HeldAtFrw
from pbh.state import State
from pbh.stencils import FaceClosure
from pbh.timestep import COURANT_NUMBER, Integrator, Scheme, advance, courant_step
from pbh.types import FloatArray

#: The first zeros of j_1, so that k = zero / X_N gives a mode with delta_U = 0 at the outer face.
J1_ZEROS = (4.493409457909064, 7.725251836937707, 10.904121659428899)


@dataclass(frozen=True)
class BesselMode:
    """The growing linear mode of wavenumber `k` and amplitude `B`."""

    k: float
    B: float

    def delta_rho(self, bg: Background, X: FloatArray) -> FloatArray:
        """`delta_rho(xi, X)`, the relative density perturbation, at the radii `X`."""
        z = self.k * bg.tau
        return bg.tau * self.k * spherical_jn(0, self.k * X) * self.B * spherical_jn(1, z)

    def delta_U(self, bg: Background, X: FloatArray) -> FloatArray:
        """`delta_U(xi, X) = U / X - 1` at the radii `X` (zero at the origin by continuity)."""
        z = self.k * bg.tau
        with np.errstate(invalid="ignore", divide="ignore"):
            profile = np.where(X > 0.0, spherical_jn(1, self.k * X) / X, self.k / 3.0)  # j_1(kX)/X -> k/3
        return -0.75 * bg.tau * profile * self.B * (spherical_jn(1, z) - z * spherical_jn(2, z))

    def cell_contents(self, bg: Background, geo: Geometry) -> FloatArray:
        """The exact energy content `int_cell X^2 (1 + delta_rho) dX` of every cell."""
        z = self.k * bg.tau
        antiderivative = geo.X**2 * spherical_jn(1, self.k * geo.X) / self.k  # int X^2 j_0(k X) dX
        return geo.dV + bg.tau * self.k * self.B * spherical_jn(1, z) * np.diff(antiderivative)

    def state(self, bg: Background, geo: Geometry) -> State:
        """The exact state of the mode: exact cell contents and face velocities."""
        return State(E=self.cell_contents(bg, geo), U=geo.X * (1.0 + self.delta_U(bg, geo.X)), W=0.0)


def mode_errors(m: Map, k_index: int, N: int, settings: KernelSettings, xi_end: float = 1.0) -> tuple[float, float]:
    """The L1 errors of the cell contents and the face velocities after evolving a mode from xi = 0 to xi_end.

    The mode's wavenumber is chosen so that delta_U vanishes at the outer face, where the held closure is then exact.
    The amplitude is small enough that the scheme's nonlinear response, of relative order B, stays below the
    truncation error being measured (at B = 1e-4 it floors the density error near 1e-5 and the rates fall off).
    """
    eos = EquationOfState(RADIATION)
    X_N = float(m.radii(0.0, N)[0][N])
    mode = BesselMode(k=J1_ZEROS[k_index] / X_N, B=1e-6)
    sch = Scheme(eos, m, Layout(N), FaceClosure.FIRST_ORDER, HeldAtFrw(), settings)
    geo = sch.frame(0.0).geo
    xi = 0.0
    dy = sch.layout.pack(mode.state(Background.at(eos, 0.0), geo)) - sch.frw(0.0)
    while xi < xi_end - 1e-12:
        res = sch.evaluate(xi, sch.frw(xi) + dy)
        dxi = min(courant_step(res, geo, sch.layout, COURANT_NUMBER), xi_end - xi)
        dy = advance(sch, Integrator.RK4, xi, dy, dxi)
        xi += dxi
    final = sch.layout.unpack(sch.frw(xi) + dy)
    exact = mode.state(Background.at(eos, xi), geo)
    err_E = float(np.sum(np.abs(final.E - exact.E)) / np.sum(np.abs(exact.E - geo.dV)))
    err_U = float(np.sum(np.abs(final.U[1:] - exact.U[1:])) / np.sum(np.abs(exact.U[1:] - geo.X[1:])))
    return err_E, err_U

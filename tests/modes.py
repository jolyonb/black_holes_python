"""One exact linear mode of radiation for the convergence tests, built on pbh.initial.GrowingMode.

Choosing the wavenumber so that `j_1(k X_N) = 0` makes `delta_U` vanish at the outer face for all time, so the held
closure of `outer.py` is exact there and the mode can be evolved without a boundary datum.
"""

import numpy as np
from evolve import evolve

from pbh.eos import RADIATION, Background, EquationOfState
from pbh.geometry import Geometry
from pbh.initial import GrowingMode
from pbh.kernels import KernelSettings
from pbh.layout import Layout
from pbh.maps import Map
from pbh.outer import HeldAtFrw
from pbh.state import State
from pbh.stencils import FaceClosure
from pbh.timestep import Scheme

#: The first zeros of j_1, so that k = zero / X_N gives a mode with delta_U = 0 at the outer face.
J1_ZEROS = (4.493409457909064, 7.725251836937707, 10.904121659428899)


def single_mode(k: float, B: float) -> GrowingMode:
    """The growing mode of one wavenumber `k` and amplitude `B`."""
    return GrowingMode(k=np.array([k]), B=np.array([B]))


def mode_state(mode: GrowingMode, bg: Background, geo: Geometry) -> State:
    """The exact state of a mode: exact cell contents and face velocities, `U_0 = 0`, no boundary scalar."""
    X = geo.X[: geo.N + 1]
    E = np.diff(X**3 * (1.0 + mode.delta_m(bg, X))) / 3.0
    U = X * (1.0 + mode.delta_U(bg, X))
    U[0] = 0.0
    return State(E=E, U=U, W=0.0)


def mode_errors(m: Map, k_index: int, N: int, settings: KernelSettings, xi_end: float = 1.0) -> tuple[float, float]:
    """The L1 errors of the cell contents and the face velocities after evolving a mode from xi = 0 to xi_end.

    The mode's wavenumber is chosen so that delta_U vanishes at the outer face, where the held closure is then exact.
    The amplitude is small enough that the scheme's nonlinear response, of relative order B, stays below the
    truncation error being measured (at B = 1e-4 it floors the density error near 1e-5 and the rates fall off).
    """
    eos = EquationOfState(RADIATION)
    X_N = float(m.radii(0.0, N)[0][N])
    mode = single_mode(J1_ZEROS[k_index] / X_N, B=1e-6)
    sch = Scheme(eos, m, Layout(N), FaceClosure.FIRST_ORDER, HeldAtFrw(), settings)
    geo = sch.frame(0.0).geo
    final = evolve(sch, mode_state(mode, Background.at(eos, 0.0), geo), 0.0, xi_end)
    exact = mode_state(mode, Background.at(eos, xi_end), geo)
    err_E = float(np.sum(np.abs(final.E - exact.E)) / np.sum(np.abs(exact.E - geo.dV)))
    err_U = float(np.sum(np.abs(final.U[1:] - exact.U[1:])) / np.sum(np.abs(exact.U[1:] - geo.X[1:])))
    return err_E, err_U

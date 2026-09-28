"""Type stubs of `pbh_engine`, the Rust engine (src/), for pyright strict. Only `pbh.rust_engine` imports it.

maturin ships this file in the wheel, with a `py.typed` marker, as the module's stubs.

Every array a `StageOutput` hands back is new to that stage call (a second read of the same attribute returns the
same object, not another copy), owned by numpy, and has its Python length: `N` for a cell field and
`N + 1` for a face field, NaN where the numpy engine has NaN (the sign bit of a NaN the stage computes is unspecified).
The packed vectors must be one-dimensional native float64 (`pbh.rust_engine.RustStage.packed` converts).
"""

from typing import final

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]  # pbh.types.FloatArray, restated so that the stubs import nothing from pbh

@final
class StageFrame:
    """One frame copied into Rust: the geometry, the stencil weights, the FRW reference and two scalar squares."""

    def __init__(
        self,
        *,
        j_e: int,
        X: FloatArray,
        X_xi: FloatArray,
        dV: FloatArray,
        sbar: FloatArray,
        dS: FloatArray,
        dX: FloatArray,
        Xm: FloatArray,
        X2: FloatArray,
        X3: FloatArray,
        s_in: FloatArray,
        s_out: FloatArray,
        grad_s: FloatArray,
        centred_U: FloatArray,
        r_L: FloatArray,
        r_R: FloatArray,
        outer_U: tuple[float, float, float],
        excision_U: float,
        outer_rho: tuple[float, float, float],
        state_E: FloatArray,
        state_U: FloatArray,
        state_M_e: float,
        rate_E: FloatArray,
        rate_U: FloatArray,
        rate_M_e: float,
        frw_speed: FloatArray,
        F_frw: FloatArray,
        X_N_squared: float,
        X_je_squared: float,
    ) -> None:
        """Copy the frame's arrays; `X_N_squared` and `X_je_squared` are the Python's `pow` of `X_N` and `X_je`."""

@final
class StageSettings:
    """The Scheme's settings: the equation of state's floats, the kernel switches (enum values) and the closure."""

    def __init__(
        self,
        *,
        w_float: float,
        alpha_float: float,
        sqrt_w: float,
        lapse_exponent: float,
        energy_source_rate: float,
        is_radiation: bool,
        kernels: str,
        density_limiter: str,
        c_v: float,
        theta: float,
        viscous_flux: str,
        cap_tension: bool,
        closure: str,
        tau_u: float | None = None,
        tau_rho: float | None = None,
        tau_W: float | None = None,
        rho_N: float | None = None,
        ephi_N: float | None = None,
    ) -> None:
        """Hold the settings.

        `closure` is `held_at_frw`, `outgoing_wave` (with the three strengths) or `held_exterior` (with `rho_N` and
        `ephi_N`); anything else, or parameters that do not match it, raises `ValueError`.
        """

@final
class DerivedOutput:
    """The fields of `pbh.derived.Derived`."""

    @property
    def rho(self) -> FloatArray:
        """`rho_c = E_c / Delta V_c` (cells)."""

    @property
    def ephi(self) -> FloatArray:
        """The cell lapse (cells)."""

    @property
    def delta_ephi(self) -> FloatArray:
        """`ephi_c - 1` (cells)."""

    @property
    def M(self) -> FloatArray:
        """The cumulative mass `M_j` (faces)."""

    @property
    def delta_M(self) -> FloatArray:
        """`M_j - X_j^3` (faces)."""

    @property
    def mt(self) -> FloatArray:
        """`M_j / X_j^3` (faces)."""

    @property
    def delta_rho(self) -> FloatArray:
        """`rho_c - 1` (cells)."""

    @property
    def delta_U(self) -> FloatArray:
        """`U_j / X_j - 1` (faces)."""

    @property
    def delta_m(self) -> FloatArray:
        """`mt_j - 1` (faces)."""

    @property
    def Gammabar2(self) -> FloatArray:
        """`Gammabar_j^2` (faces)."""

    @property
    def rho_f(self) -> FloatArray:
        """`<rho>_j` (faces)."""

    @property
    def ephi_f(self) -> FloatArray:
        """`<ephi>_j` (faces)."""

    @property
    def delta_rho_f(self) -> FloatArray:
        """`<rho>_j - 1` (faces)."""

    @property
    def delta_ephi_f(self) -> FloatArray:
        """`<ephi>_j - 1` (faces)."""

@final
class SpeedsOutput:
    """The fields of `pbh.equations.Speeds`."""

    @property
    def drift(self) -> FloatArray:
        """`alpha (<ephi>_j U_j - h X_j)` (faces)."""

    @property
    def Theta(self) -> FloatArray:
        """The grid velocity `Theta_j` (faces)."""

    @property
    def cE(self) -> FloatArray:
        """The energy-flux velocity `cE_j` (faces)."""

    @property
    def a(self) -> FloatArray:
        """The sound speed `a_j` (faces)."""

    @property
    def Lam(self) -> FloatArray:
        """The signal speed `Lambda_j` (faces)."""

@final
class KernelOutput:
    """The fields of `pbh.kernels.KernelResult`."""

    @property
    def rho_L(self) -> FloatArray:
        """The density reconstructed to each face from inside (faces)."""

    @property
    def rho_R(self) -> FloatArray:
        """... from outside (faces)."""

    @property
    def delta_rho_L(self) -> FloatArray:
        """`rho_L - 1` (faces)."""

    @property
    def delta_rho_R(self) -> FloatArray:
        """`rho_R - 1` (faces)."""

    @property
    def J(self) -> FloatArray:
        """The limited velocity jump (cells)."""

    @property
    def q(self) -> FloatArray:
        """The viscous pressure (cells)."""

    @property
    def q_f(self) -> FloatArray:
        """Its face value (faces)."""

    @property
    def Q(self) -> FloatArray:
        """Its areal force (faces)."""

    @property
    def F(self) -> FloatArray:
        """The HLL flux, NaN at face `N` (faces)."""

    @property
    def theta_scale(self) -> FloatArray:
        """The theta-limiter's factor (cells)."""

    @property
    def Lam_plus(self) -> FloatArray:
        """`Lambda^+_j` (faces)."""

    @property
    def Lam_minus(self) -> FloatArray:
        """`Lambda^-_j` (faces)."""

    @property
    def v_L(self) -> FloatArray:
        """The inside chord speed (faces)."""

    @property
    def v_R(self) -> FloatArray:
        """The outside chord speed (faces)."""

@final
class StageOutput:
    """Everything one stage computed: the fields of `pbh.equations.DerivsResult`."""

    @property
    def rate_E(self) -> FloatArray:
        """The rate of the cell energies (cells)."""

    @property
    def rate_U(self) -> FloatArray:
        """The rate of the face velocities (faces)."""

    @property
    def rate_W(self) -> float:
        """The rate of `W`."""

    @property
    def rate_M_e(self) -> float:
        """The rate of `M_e`."""

    @property
    def deviation_rate_E(self) -> FloatArray:
        """The rate of the energies' deviation from FRW (cells)."""

    @property
    def deviation_rate_U(self) -> FloatArray:
        """The rate of the velocities' deviation (faces)."""

    @property
    def deviation_rate_W(self) -> float:
        """The rate of `W` (its FRW value is zero)."""

    @property
    def deviation_rate_M_e(self) -> float:
        """The rate of `M_e`'s deviation."""

    @property
    def derived(self) -> DerivedOutput:
        """The derived fields."""

    @property
    def speeds(self) -> SpeedsOutput:
        """The speeds."""

    @property
    def F(self) -> FloatArray:
        """The energy flux through every retained face (faces)."""

    @property
    def delta_F(self) -> FloatArray:
        """The flux less its FRW value (faces)."""

    @property
    def kernels(self) -> KernelOutput | None:
        """The kernels' fields, or `None` when the centred base scheme ran."""

def stage_deviation(
    frame: StageFrame, settings: StageSettings, Gammabar2: float, c_s: float, hubble: float, dy: FloatArray
) -> StageOutput:
    """One stage at `y_FRW + delta y` from the packed deviation (`Scheme.evaluate_deviation`).

    Raises:
        NotHyperbolicError: As `derive` raises it.
        ValueError: As the outer closure raises it, or if `dy` has not the packed length (`Layout.check_packed`).
        TypeError: If `dy` is not a one-dimensional native float64 array.
    """

def stage_state(
    frame: StageFrame, settings: StageSettings, Gammabar2: float, c_s: float, hubble: float, y: FloatArray
) -> StageOutput:
    """One stage at the packed whole state `y` (`Scheme.evaluate`); raises as `stage_deviation` does."""

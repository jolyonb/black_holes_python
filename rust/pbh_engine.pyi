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
    """One frame built in Rust from the map at the faces: the geometry, the stencil weights and the FRW reference.

    The getters hand each array back as a fresh numpy array: those of `Geometry`, `StencilWeights` and
    `FrwReference` (the reference's state and rate as `state_E`, `state_U`, `state_M_e` and `rate_E`, `rate_U`,
    `rate_M_e`), formed as the Python forms them.
    """

    def __init__(self, settings: StageSettings, *, j_e: int, X: FloatArray, X_xi: FloatArray, hubble: float) -> None:
        """Build the frame from `X_j` and `(d_xi X)_j` at the `N + 2` faces `0..N+1` (`Map.radii`)."""
    @property
    def X(self) -> FloatArray: ...
    @property
    def X_xi(self) -> FloatArray: ...
    @property
    def dV(self) -> FloatArray: ...
    @property
    def dV_xi(self) -> FloatArray: ...
    @property
    def sbar(self) -> FloatArray: ...
    @property
    def dS(self) -> FloatArray: ...
    @property
    def dX(self) -> FloatArray: ...
    @property
    def Xm(self) -> FloatArray: ...
    @property
    def X2(self) -> FloatArray: ...
    @property
    def X3(self) -> FloatArray: ...
    @property
    def s_in(self) -> FloatArray: ...
    @property
    def s_out(self) -> FloatArray: ...
    @property
    def grad_s(self) -> FloatArray: ...
    @property
    def centred_U(self) -> FloatArray: ...
    @property
    def r_L(self) -> FloatArray: ...
    @property
    def r_R(self) -> FloatArray: ...
    @property
    def outer_U(self) -> tuple[float, float, float]: ...
    @property
    def excision_U(self) -> float: ...
    @property
    def outer_rho(self) -> tuple[float, float, float]: ...
    @property
    def state_E(self) -> FloatArray: ...
    @property
    def state_U(self) -> FloatArray: ...
    @property
    def state_M_e(self) -> float: ...
    @property
    def rate_E(self) -> FloatArray: ...
    @property
    def rate_U(self) -> FloatArray: ...
    @property
    def rate_M_e(self) -> float: ...
    @property
    def frw_speed(self) -> FloatArray: ...
    @property
    def F_frw(self) -> FloatArray: ...

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

@final
class AttemptOutput:
    """One checked attempt's outcome (`pbh.timestep.Attempt`)."""

    @property
    def stages(self) -> list[tuple[float, float, float, float, float, FloatArray]]:
        """Every stage after the first that completed: `(F_N, F_je, M_total, delta_F_N, delta_M_total, k)`."""

    @property
    def dy(self) -> FloatArray | None:
        """The deviation arrived at, or `None` if the attempt was refused."""

    @property
    def result(self) -> StageOutput | None:
        """The stage at the deviation arrived at, or `None` if the attempt was refused."""

    @property
    def failure(self) -> tuple[str, int, int, float] | None:
        """The refusal `(cause, stage, index, value)`, `cause` a `FailureCause` value, or `None`."""

def checked_step(
    settings: StageSettings,
    frames: list[StageFrame],
    backgrounds: list[tuple[float, float, float]],
    arrive: StageFrame,
    arrive_background: tuple[float, float, float],
    dy: FloatArray,
    dxi: float,
    k1: FloatArray,
    a: list[list[float]],
    b: list[float],
) -> AttemptOutput:
    """One checked attempt (`timestep.checked_step`) from `dy` and the first stage's packed rate `k1`.

    The stages after the first are evaluated on `frames`, each with its background `(Gammabar2, c_s, hubble)`, and the
    result on `arrive`.

    Raises:
        ValueError: As the outer closure raises it, or if the frames, backgrounds, tableau or vectors do not match.
        TypeError: If a vector is not a one-dimensional native float64 array.
    """

def blend_radii(
    B: FloatArray, weights: FloatArray, pinned: list[float], rates: list[float], alpha: float
) -> tuple[FloatArray, FloatArray]:
    """`BlendMap.radii` from the map's static part (`B` and the rows of `weights`) and each zone's ramp scalars.

    Raises:
        ValueError: If the rows, their lengths or the rates do not match.
    """

def emptying_rates(
    Lam_plus: FloatArray,
    Lam_minus: FloatArray,
    v_L: FloatArray,
    v_R: FloatArray,
    rho_L: FloatArray,
    rho_R: FloatArray,
    X2: FloatArray,
    E: FloatArray,
    F_N: float,
    energy_source_rate: float,
    j_e: int,
) -> FloatArray:
    """Each retained cell's emptying rate (`monitors.emptying_rates`, with the kernels on); NaN below `j_e`.

    Raises:
        ValueError: If a face field is not one longer than the cell energies, or `j_e > N - 2`.
    """

def near_zone(
    Xm: FloatArray,
    X: FloatArray,
    ephi: FloatArray,
    rho: FloatArray,
    U: FloatArray,
    Gammabar2: FloatArray,
    radii: list[float],
) -> tuple[list[float], int]:
    """`horizon.near_zone_numbers`: the near-zone monitors at each radius, and the retained cell of least lapse.

    Raises:
        ValueError: If the cell and face arrays do not match.
    """

def radii_admissible(X: FloatArray) -> bool:
    """Whether the radii are a map's (`geometry.check_radii`): `X_0 = 0` and every `X_(j+1) - X_j > 0`."""

@final
class TrappingOutput:
    """The horizon finder's numbers on one slice (`pbh.horizon.Trapping`)."""

    @property
    def h(self) -> FloatArray:
        """The trapping function `U + Gammabar` (faces), NaN below the excision face."""

    @property
    def trapped_faces(self) -> int:
        """How many retained faces are trapped."""

    @property
    def crossings(self) -> list[tuple[int, float, bool]]:
        """Every sign change `(j, t, outer)`, from the origin outward."""

    @property
    def margin(self) -> float:
        """The smallest `1 + U / Gammabar` over the retained faces."""

    @property
    def margin_face(self) -> int:
        """Where."""

    @property
    def core_margin(self) -> float:
        """The same over the central infall region."""

    @property
    def core_margin_face(self) -> int:
        """Where."""

    @property
    def outer_face_trapped(self) -> bool:
        """Whether face `N` is trapped."""

def trapping(U: FloatArray, Gammabar2: FloatArray, j_e: int) -> TrappingOutput:
    """The trapping function on the retained faces `j_e..N` and what the finder reads from it (`horizon.trapping`).

    Raises:
        ValueError: If the arrays differ in length or are shorter than three, or `j_e > N - 2`.
        TypeError: If either is not a one-dimensional native float64 array.
    """

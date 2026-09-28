"""The adapter between a `Scheme` and the Rust engine `pbh_engine` (the switch `numerics: {engine: rust}`).

The Rust engine computes one stage, `equations.calc_derivs` and everything it calls, with the same operations in the
same order, one Rust function per Python function (`rust/src`). It is a separate, optional package (`pbh-engine` in
`rust/`, the dependency group `rust`), and `pbh` runs without it. This module is the only Python that imports it, and
only a `Scheme` built with `Engine.RUST` imports this module, so that `import pbh` and the numpy engine never load the
extension, and a Scheme asked for the Rust engine where it is not installed says how to install it. It does three
things:

* once per `Scheme`, `RustStage` hands the extension the settings: the floats of the equation of state, the kernel
  switches, and the outer closure with its parameters. Only the three closures of `outer.py` have Rust rows, and a
  closure is matched by its exact type, so that a subclass cannot silently run its parent's rows;
* once per frame, `RustStage.frame` copies the frame's geometry, stencil weights and FRW reference into Rust, with the
  two scalar squares the Python forms by the C library's `pow` (`X_N ** 2` of the closures, `X[j_e] ** 2` of the
  viscous pressure's excision row; see `Geometry`), formed here by the same expressions, so that Rust never forms
  them itself. The one power Rust does form is the lapse of a general `w` (neither radiation nor the stiff fluid),
  `rho ** lapse_exponent` with `expm1` and `log1p`: Rust calls the platform's C library there, which is what numpy
  calls on macOS, but numpy may use its own SIMD versions elsewhere, so for such a `w` the engines can differ there by
  an ulp or so (`rust/src/numpy_like.rs`);
* per stage, it passes the packed deviation (or the packed state), as native float64 after `Layout.unpack`'s shape
  check, and the background scalars, and assembles the returned arrays into the `DerivsResult`, `Derived`, `Speeds`,
  `KernelResult` and `State`s that every consumer reads.

A stage outside the hyperbolic domain raises `derived.NotHyperbolicError` from inside Rust, with the same field, index
and value; a closure's refusal raises `ValueError` with the same message, at the same point of the stage.

What the engines return agrees to the bit on every non-NaN entry, and the NaN entries are the same entries
(tests/test_rust_engine.py): at `w` = 1/3 and 1 by construction, the stage using only correctly rounded operations
there, and at every other `w` on the machine it was compared on (macOS on arm64; see the lapse above). Two things
differ, and neither reaches a decision, since the driver tests finiteness only:

* the sign bit of a NaN that the stage computes (a NaN merely copied from the input or from a NaN fill is the same):
  numpy applies every unary minus literally, and the compiler may fold a negation into a neighbouring subtraction
  (`-(a - b) + c` as `c - (a - b)`), which is exact for every number but leaves a NaN's sign as it was;
* warnings: numpy emits a `RuntimeWarning` on overflow or an invalid operation in any array operation (an
  overflowing deviation, for one, warns from `derived.gammabar_squared` before `derive` refuses the state; so does
  the branch of an `np.where` that is not taken), and Rust emits none. Where warnings are errors, as under this
  package's pytest settings or `python -W error`, such a state therefore raises the warning on the numpy engine and
  goes on to its result or refusal on the Rust engine, which is what numpy gives with its warnings silenced
  (`np.errstate`). A run on the Rust engine logs none of these warnings.
"""

import numpy as np
import pbh_engine
from pbh_engine import StageFrame, StageOutput, StageSettings

from pbh.derived import Derived
from pbh.eos import Background, EquationOfState
from pbh.equations import DerivsResult, Speeds
from pbh.geometry import Geometry
from pbh.kernels import KernelResult, KernelSettings
from pbh.layout import Layout
from pbh.outer import HeldAtFrw, HeldExterior, OuterClosure, OutgoingWave
from pbh.state import FrwReference, State
from pbh.stencils import StencilWeights
from pbh.types import FloatArray


class RustStage:
    """A Scheme's stage on the Rust engine: the settings, held once, and the frames and stages built from them."""

    def __init__(self, eos: EquationOfState, settings: KernelSettings, outer: OuterClosure, layout: Layout) -> None:
        """Hand the extension the settings of the stage.

        Raises:
            ValueError: If the closure is not exactly one of `HeldAtFrw`, `OutgoingWave` or `HeldExterior`.
        """
        tau_u = tau_rho = tau_W = rho_N = ephi_N = None
        if type(outer) is HeldAtFrw:
            closure = "held_at_frw"
        elif type(outer) is OutgoingWave:
            closure = "outgoing_wave"
            tau_u, tau_rho, tau_W = outer.strengths.tau_u, outer.strengths.tau_rho, outer.strengths.tau_W
        elif type(outer) is HeldExterior:
            closure = "held_exterior"
            rho_N, ephi_N = outer.rho_N, outer.ephi_N
        else:
            raise ValueError(f"the rust engine has no rows for {outer!r}")
        self._layout = layout
        self._settings = StageSettings(
            w_float=eos.w_float,
            alpha_float=eos.alpha_float,
            sqrt_w=eos.sqrt_w,
            lapse_exponent=eos.lapse_exponent,
            energy_source_rate=eos.energy_source_rate,
            is_radiation=eos.is_radiation,
            kernels=settings.kernels.value,
            density_limiter=settings.density_limiter.value,
            c_v=settings.c_v,
            theta=settings.theta,
            viscous_flux=settings.viscous_flux.value,
            cap_tension=settings.cap_tension,
            closure=closure,
            tau_u=tau_u,
            tau_rho=tau_rho,
            tau_W=tau_W,
            rho_N=rho_N,
            ephi_N=ephi_N,
        )

    def frame(self, geo: Geometry, w: StencilWeights, reference: FrwReference) -> StageFrame:
        """Copy one frame into Rust: its geometry, stencil weights and FRW reference, and the two scalar squares."""
        N, j_e = geo.N, self._layout.j_e
        return StageFrame(
            j_e=j_e,
            X=geo.X,
            X_xi=geo.X_xi,
            dV=geo.dV,
            sbar=geo.sbar,
            dS=geo.dS,
            dX=geo.dX,
            Xm=geo.Xm,
            X2=geo.X2,
            X3=geo.X3,
            s_in=geo.s_in,
            s_out=geo.s_out,
            grad_s=w.grad_s,
            centred_U=w.centred_U,
            r_L=w.r_L,
            r_R=w.r_R,
            outer_U=w.outer_U,
            excision_U=w.excision_U,
            outer_rho=w.outer_rho,
            state_E=reference.state.E,
            state_U=reference.state.U,
            state_M_e=reference.state.M_e,
            rate_E=reference.rate.E,
            rate_U=reference.rate.U,
            rate_M_e=reference.rate.M_e,
            frw_speed=reference.frw_speed,
            F_frw=reference.F_frw,
            X_N_squared=float(geo.X[N]) ** 2,  # the closures' `X_N**2` on the float X_N
            X_je_squared=float(geo.X[j_e] ** 2),  # `kernels.viscous_pressure`'s `X[j_e] ** 2` on the numpy scalar
        )

    def packed(self, y: FloatArray) -> FloatArray:
        """The packed vector as the extension takes it: native float64, after `Layout.unpack`'s shape check.

        The extension takes only a one-dimensional native float64 array. The numpy engine also accepts other dtypes
        (integers, float32, big-endian float64), which `unpack` converts exactly on assigning into its float64 arrays;
        converting here first gives the same doubles, and leaves a native float64 vector as it is, uncopied.

        Raises:
            ValueError: As `Layout.unpack` raises it, with the same message.
        """
        self._layout.check_packed(y)
        return np.asarray(y, dtype=np.float64)

    def evaluate_deviation(self, frame: StageFrame, bg: Background, dy: FloatArray) -> DerivsResult:
        """`Scheme.evaluate_deviation` on the Rust engine: the stage at `y_FRW + delta y` from the packed deviation."""
        dy = self.packed(dy)
        return to_result(pbh_engine.stage_deviation(frame, self._settings, bg.Gammabar2, bg.c_s, bg.hubble, dy))

    def evaluate(self, frame: StageFrame, bg: Background, y: FloatArray) -> DerivsResult:
        """`Scheme.evaluate` on the Rust engine: the stage at the packed whole state `y`."""
        y = self.packed(y)
        return to_result(pbh_engine.stage_state(frame, self._settings, bg.Gammabar2, bg.c_s, bg.hubble, y))


def to_result(out: StageOutput) -> DerivsResult:
    """The `DerivsResult` of a Rust stage, assembled from its arrays: the same dataclasses the numpy engine returns."""
    d = out.derived
    derived = Derived(
        rho=d.rho,
        ephi=d.ephi,
        delta_ephi=d.delta_ephi,
        M=d.M,
        delta_M=d.delta_M,
        mt=d.mt,
        delta_rho=d.delta_rho,
        delta_U=d.delta_U,
        delta_m=d.delta_m,
        Gammabar2=d.Gammabar2,
        rho_f=d.rho_f,
        ephi_f=d.ephi_f,
        delta_rho_f=d.delta_rho_f,
        delta_ephi_f=d.delta_ephi_f,
    )
    s = out.speeds
    speeds = Speeds(drift=s.drift, Theta=s.Theta, cE=s.cE, a=s.a, Lam=s.Lam)
    k = out.kernels
    kernels = None
    if k is not None:
        kernels = KernelResult(
            rho_L=k.rho_L,
            rho_R=k.rho_R,
            delta_rho_L=k.delta_rho_L,
            delta_rho_R=k.delta_rho_R,
            J=k.J,
            q=k.q,
            q_f=k.q_f,
            Q=k.Q,
            F=k.F,
            theta_scale=k.theta_scale,
            Lam_plus=k.Lam_plus,
            Lam_minus=k.Lam_minus,
            v_L=k.v_L,
            v_R=k.v_R,
        )
    return DerivsResult(
        rate=State(E=out.rate_E, U=out.rate_U, W=out.rate_W, M_e=out.rate_M_e),
        deviation_rate=State(
            E=out.deviation_rate_E, U=out.deviation_rate_U, W=out.deviation_rate_W, M_e=out.deviation_rate_M_e
        ),
        derived=derived,
        speeds=speeds,
        F=out.F,
        delta_F=out.delta_F,
        kernels=kernels,
    )

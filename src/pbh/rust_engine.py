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
* once per frame, `RustStage.frame` hands Rust the map at the faces, and Rust builds the frame: the geometry, the
  stencil weights and the FRW reference (`Geometry.of`, `StencilWeights.of`, `FrwReference.of`, the same operations in
  the same order), with the scalar powers the Python takes by the C library's `pow` (`X_{j_e} ** 3` and `** 2` of the
  reference, `X_N ** 2` of the closures, `X[j_e] ** 2` of the viscous pressure's excision row; see `Geometry`) taken by
  that same `pow`. The Python's `Frame` then holds Rust's numbers, read back as numpy arrays. Every other power Rust
  forms is the lapse of a general `w` (neither radiation nor the stiff fluid), `rho ** lapse_exponent` with `expm1`
  and `log1p`: Rust calls the platform's C library there, which is what numpy calls on macOS, but numpy may use its
  own SIMD versions elsewhere, so for such a `w` the engines can differ there by an ulp or so
  (`rust/src/numpy_like.rs`);
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
from pbh.geometry import Geometry, check_radii
from pbh.horizon import Trapping
from pbh.kernels import KernelResult, KernelSettings
from pbh.layout import Layout
from pbh.outer import HeldAtFrw, HeldExterior, OuterClosure, OutgoingWave
from pbh.state import FrwReference, State
from pbh.stencils import StencilWeights
from pbh.types import FloatArray, read_only


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
        self._eos = eos
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

    def frame(
        self, X: FloatArray, X_xi: FloatArray, hubble: float
    ) -> tuple[Geometry, StencilWeights, FrwReference, StageFrame]:
        """Build one frame in Rust from the map at the `N + 2` faces, and the Python's objects from its arrays.

        The Rust frame is `Geometry.of`, `StencilWeights.of` and `FrwReference.of` with the same operations in the same
        order (tests/test_rust_engine.py asserts every array equal); the `Geometry`, `StencilWeights` and
        `FrwReference` returned hold its numbers, for the finder, the monitors and the step rules to read. The radii
        are checked here as `Geometry.of` checks them.

        Raises:
            ValueError: As `Geometry.of` raises it, with the same message.
        """
        check_radii(X)
        f = StageFrame(self._settings, j_e=self._layout.j_e, X=X, X_xi=X_xi, hubble=hubble)
        geo = Geometry(
            X=f.X,
            X_xi=f.X_xi,
            dV=f.dV,
            dV_xi=f.dV_xi,
            sbar=f.sbar,
            dS=f.dS,
            dX=f.dX,
            Xm=f.Xm,
            X2=f.X2,
            X3=f.X3,
            s_in=f.s_in,
            s_out=f.s_out,
        )
        w = StencilWeights(
            layout=self._layout,
            grad_s=f.grad_s,
            centred_U=f.centred_U,
            outer_U=f.outer_U,
            excision_U=f.excision_U,
            outer_rho=f.outer_rho,
            r_L=f.r_L,
            r_R=f.r_R,
        )
        reference = FrwReference(
            geo=geo,
            eos=self._eos,
            j_e=self._layout.j_e,
            hubble=hubble,
            state=State(E=f.state_E, U=f.state_U, W=0.0, M_e=f.state_M_e),
            rate=State(E=f.rate_E, U=f.rate_U, W=0.0, M_e=f.rate_M_e),
            frw_speed=f.frw_speed,
            F_frw=f.F_frw,
        )
        read_only(
            reference.state.E,
            reference.state.U,
            reference.rate.E,
            reference.rate.U,
            reference.frw_speed,
            reference.F_frw,
        )
        return geo, w, reference, f

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


def trapping(U: FloatArray, Gammabar2: FloatArray, layout: Layout) -> Trapping:
    """`horizon.trapping` on the Rust engine: the same numbers, by the same operations (`rust/src/horizon.rs`)."""
    out = pbh_engine.trapping(np.asarray(U, dtype=np.float64), np.asarray(Gammabar2, dtype=np.float64), layout.j_e)
    return Trapping(
        h=out.h,
        trapped_faces=out.trapped_faces,
        crossings=tuple(out.crossings),
        margin=out.margin,
        margin_face=out.margin_face,
        core_margin=out.core_margin,
        core_margin_face=out.core_margin_face,
        outer_face_trapped=out.outer_face_trapped,
    )

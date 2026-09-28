//! One stage of the integrator: the semi-discrete equations (`pbh/equations.py`; paper Section 7.3, the face-mass law
//! eq:numbh:mass), every row formed as its deviation from the FRW rate.
//!
//! In the Python's order: the derived fields, with their hyperbolicity check; the four speeds of eq:num:facefields;
//! the flux, from the kernels of Section 7.7 or the centred base flux of eq:num:energy; the outer closure's rows; the
//! velocity rows eq:num:velocity; the energy rows eq:num:energy and the face-mass row eq:numbh:mass:
//!
//! ```text
//!     energy    d_xi E_c   = d_xi Delta V_c - (delta F_{c+1} - delta F_c) + h (2 - 3 alpha) delta E_c,
//!     face mass d_xi M_e   = d_xi X_je^3 - 3 delta F_je + h (2 - 3 alpha) delta M_e,
//!     velocity  d_xi U_j   = h (d_xi X)_j + h (1 - alpha) (U_j - h X_j) + pressure
//!                          - h alpha / 2 X_j [<ephi - 1>_j (mt_j + 3 w <rho>_j) + delta_m,j + 3 w <rho - 1>_j]
//!                          - drift_j (h + delta D_j) + (d_xi X)_j delta D_j,
//! ```
//!
//! with the background coefficient `h` (1 on FRW, 0 in flat spacetime) on every Hubble, gravity and source part,
//! `U_j - h X_j` the velocity's absolute deviation (`deviation.U[j]`, not `Derived::delta_U`, which is divided by
//! `X_j`), `drift_j = alpha (<ephi>_j U_j - h X_j)` and `delta D_j = (D_U U)_j - h`.

use crate::derived::{Derived, NotHyperbolic, derive};
use crate::eos::{Background, EquationOfState};
use crate::geometry::Geometry;
use crate::kernels::{
    KernelResult, KernelSettings, Kernels, hll_flux, reconstruct_density, viscous_pressure, viscous_sides,
};
use crate::outer::{OuterClosure, OuterInputs};
use crate::state::{FrwReference, State};
use crate::stencils::StencilWeights;

/// Why a stage could not be evaluated: the `NotHyperbolicError` of `derive`, or a closure's `ValueError`.
pub enum StageError {
    /// A non-positive density or `Gammabar^2`.
    NotHyperbolic(NotHyperbolic),
    /// The outer closure's refusal, with the Python's message.
    Closure(String),
}

/// The four speeds of eq:num:facefields at the retained faces, NaN elsewhere (`Speeds`).
pub struct Speeds {
    /// `alpha (<ephi>_j U_j - h X_j)`, the fluid's velocity relative to the Hubble flow.
    pub drift: Vec<f64>,
    /// `Theta_j = drift_j - (d_xi X)_j`, the fluid's velocity relative to the moving face.
    pub Theta: Vec<f64>,
    /// `cE_j = alpha w h X_j + (1 + w) drift_j`, the energy-flux velocity.
    pub cE: Vec<f64>,
    /// `a_j = alpha sqrt(w) <ephi>_j Gammabar_j`, the sound speed.
    pub a: Vec<f64>,
    /// `Lambda_j = |Theta_j| + a_j`, the signal speed.
    pub Lam: Vec<f64>,
}

/// What one evaluation computed (`DerivsResult`).
pub struct DerivsResult {
    /// The time derivative of the state: the FRW rate plus `deviation_rate`.
    pub rate: State,
    /// The time derivative of the deviation from FRW, which the integrator advances.
    pub deviation_rate: State,
    /// The derived fields of this stage.
    pub derived: Derived,
    /// The speeds of this stage.
    pub speeds: Speeds,
    /// The energy flux through every retained face.
    pub F: Vec<f64>,
    /// The same flux less its FRW value.
    pub delta_F: Vec<f64>,
    /// What the shock-capturing kernels produced, or `None` when the centred base scheme ran.
    pub kernels: Option<KernelResult>,
}

/// The four speeds of eq:num:facefields from the derived fields and the deviation, by way of the drift (`speeds`).
///
/// `Theta`, `cE` and `Lam` are formed over the whole array, as numpy forms them, so their NaN entries come out of the
/// same arithmetic.
pub fn speeds(
    d: &Derived,
    deviation: &State,
    geo: &Geometry,
    eos: &EquationOfState,
    j_e: usize,
    hubble: f64,
) -> Speeds {
    let alpha = eos.alpha_float;
    let w = eos.w_float;
    let N = geo.dV.len();
    let mut drift = vec![f64::NAN; N + 1]; // NaN below the retained faces, and so are Theta and cE
    let mut a = vec![f64::NAN; N + 1];
    for j in j_e..N + 1 {
        drift[j] = alpha * (hubble * geo.X[j] * d.delta_ephi_f[j] + d.ephi_f[j] * deviation.U[j]); // <ephi> U - h X
        a[j] = alpha * eos.sqrt_w * d.ephi_f[j] * d.Gammabar2[j].sqrt();
    }
    let mut Theta = vec![0.0; N + 1];
    let mut cE = vec![0.0; N + 1];
    let mut Lam = vec![0.0; N + 1];
    for j in 0..N + 1 {
        Theta[j] = drift[j] - geo.X_xi[j];
        cE[j] = alpha * w * hubble * geo.X[j] + (1.0 + w) * drift[j];
        Lam[j] = Theta[j].abs() + a[j];
    }
    Speeds {
        drift,
        Theta,
        cE,
        a,
        Lam,
    }
}

/// Evaluate the semi-discrete equations once: the rate of every unknown at this time and state (`calc_derivs`, with
/// the caller's deviation and reference, as `Scheme.evaluate` and `Scheme.evaluate_deviation` hand them).
///
/// `X_N_squared` and `X_je_squared` are the two scalar squares the Python forms by the C library's `pow`: `X_N ** 2`
/// of the closures and `X[j_e] ** 2` of the viscous pressure's excision row.
pub fn calc_derivs(
    state: &State,
    geo: &Geometry,
    bg: &Background,
    eos: &EquationOfState,
    w: &StencilWeights,
    outer: &OuterClosure,
    settings: &KernelSettings,
    deviation: &State,
    reference: &FrwReference,
    X_N_squared: f64,
    X_je_squared: f64,
) -> Result<DerivsResult, StageError> {
    let N = w.layout.N;
    let j_e = w.layout.j_e;
    let alpha = eos.alpha_float;
    let w_eos = eos.w_float;
    let h = bg.hubble; // 1 on FRW, 0 in flat spacetime: the coefficient of every Hubble, gravity and source term

    let d = derive(state, geo, bg, eos, w, settings.theta, deviation).map_err(StageError::NotHyperbolic)?;
    let sp = speeds(&d, deviation, geo, eos, j_e, h);
    let D_s_rho = w.gradient_s(&d.delta_rho); // the same difference as of rho, without its rounding to the FRW size
    let delta_D = w.velocity_gradient(&deviation.U); // (D_U U)_j - h: every row of D_U gives exactly h on U = h X

    // The energy flux through the retained faces and the artificial pressure force, as the flux's deviation from the
    // FRW flux: from the kernels of Section 7.7, or, with the kernels off, the centred base flux of eq:num:energy,
    // (cE_j - (d_xi X)_j) X_j^2 <rho>_j, and no force.
    let X = &geo.X;
    let X_xi = &geo.X_xi;
    let F_frw = &reference.F_frw; // the FRW flux, to which the deviation is added for the whole flux
    let frw_speed = &reference.frw_speed; // alpha w h X - d_xi X: the FRW flux is frw_speed X^2
    let (mut delta_F, Q, kernels) = match settings.kernels {
        Kernels::Production => {
            let recon = reconstruct_density(&d.delta_rho, geo, w, &settings.density_limiter, settings.theta);
            let visc = viscous_pressure(
                geo,
                &d,
                &sp.Lam,
                eos,
                w,
                settings.c_v,
                settings.cap_tension,
                X_je_squared,
            );
            let sides = viscous_sides(
                &visc.q,
                &visc.q_f,
                &d.rho,
                &recon.rho_L,
                &recon.rho_R,
                &w.layout,
                &settings.viscous_flux,
            );
            let hll = hll_flux(
                &recon.rho_L,
                &recon.rho_R,
                &recon.delta_rho_L,
                &recon.delta_rho_R,
                &sides.q_L,
                &sides.q_R,
                deviation,
                geo,
                &sp.Theta,
                &sp.a,
                eos,
                w,
                h,
                frw_speed,
            );
            // The kernels' own flux is formed before face N is set, so its face N is NaN, as in the Python.
            let mut kernel_F = vec![0.0; N + 1];
            for j in 0..N + 1 {
                kernel_F[j] = F_frw[j] + hll.delta_F[j];
            }
            let Q = visc.Q.clone();
            let kernels = KernelResult {
                rho_L: recon.rho_L,
                rho_R: recon.rho_R,
                delta_rho_L: recon.delta_rho_L,
                delta_rho_R: recon.delta_rho_R,
                J: visc.J,
                q: visc.q,
                q_f: visc.q_f,
                Q: visc.Q,
                F: kernel_F,
                theta_scale: recon.t,
                Lam_plus: hll.Lam_plus,
                Lam_minus: hll.Lam_minus,
                v_L: hll.v_L,
                v_R: hll.v_R,
            };
            (hll.delta_F, Q, Some(kernels))
        }
        Kernels::Centred => {
            let mut delta_F = vec![f64::NAN; N + 1];
            for j in j_e..N + 1 {
                // outer.base_flux_deviation, on the arrays
                delta_F[j] = geo.X2[j] * (frw_speed[j] * d.delta_rho_f[j] + (1.0 + w_eos) * sp.drift[j] * d.rho_f[j]);
            }
            if j_e == 0 {
                delta_F[0] = 0.0;
            }
            (delta_F, vec![0.0; N + 1], None) // no artificial pressure force
        }
    };

    // The outer face: the closure supplies the rows the interior cannot.
    let inputs = OuterInputs {
        X_N: X[N],
        X_N_squared,
        X_xi_N: X_xi[N],
        U_N: state.U[N],
        W: state.W,
        delta_U_N: d.delta_U[N],
        delta_rho_N_1: d.delta_rho[N - 1],
        rho_f_N: d.rho_f[N],
        delta_rho_f_N: d.delta_rho_f[N],
        ephi_f_N: d.ephi_f[N],
        delta_ephi_f_N: d.delta_ephi_f[N],
        mt_N: d.mt[N],
        delta_m_N: d.delta_m[N],
        drift_N: sp.drift[N],
        delta_DU_N: delta_D[N],
        dS_N: geo.dS[N],
        c_s: bg.c_s,
        hubble: h,
    };
    let rows = outer.rows(&inputs, eos).map_err(StageError::Closure)?;
    delta_F[N] = rows.delta_F_N;
    let mut F = vec![0.0; N + 1]; // the whole flux, for the monitors
    for j in 0..N + 1 {
        F[j] = F_frw[j] + delta_F[j];
    }

    // The velocity rows at the interior faces, eq:num:velocity, as their deviation from the FRW rate (d_xi X)_j.
    let mut dU = vec![f64::NAN; N + 1];
    for j in j_e.max(1)..N {
        let expansion = (1.0 - alpha) * h * deviation.U[j]; // h (1 - alpha) U_j, less the FRW part
        // alpha / (1 + w) <ephi> Gammabar^2 / <rho>
        let inertia = alpha / (1.0 + w_eos) * d.ephi_f[j] * d.Gammabar2[j] / d.rho_f[j];
        let pressure = -inertia * (w_eos * D_s_rho[j] + Q[j]); // ... times [w (D_s rho)_j + Q_j], the pressure force
        let lapse_mass =
            d.delta_ephi_f[j] * (d.mt[j] + 3.0 * w_eos * d.rho_f[j]) + d.delta_m[j] + 3.0 * w_eos * d.delta_rho_f[j];
        let gravity = -0.5 * alpha * h * X[j] * lapse_mass; // h alpha / 2 <ephi> X (mt + 3 w <rho>), less FRW's
        let advection = -sp.drift[j] * (h + delta_D[j]) + X_xi[j] * delta_D[j]; // Theta (D_U U), less its FRW value
        dU[j] = expansion + pressure + gravity + advection;
    }
    dU[N] = rows.delta_dU_N;
    if j_e == 0 {
        dU[0] = 0.0; // not an unknown: U_0 = 0 always
    }

    // The energy rows, eq:num:energy, and the face-mass row, eq:numbh:mass, with the same flux F_{j_e}. The FRW parts
    // cancel exactly in the algebra, -(F_FRW,c+1 - F_FRW,c) + (2 - 3 alpha) Delta V_c = d_xi Delta V_c.
    let mut dE = vec![f64::NAN; N];
    for c in j_e..N {
        let flux_out = delta_F[c + 1];
        let flux_in = delta_F[c];
        dE[c] = -(flux_out - flux_in) + h * eos.energy_source_rate * deviation.E[c];
    }
    let dM_e = if j_e > 0 {
        h * eos.energy_source_rate * deviation.M_e - 3.0 * delta_F[j_e]
    } else {
        0.0
    };

    let frw = &reference.rate;
    let mut rate_E = vec![0.0; N];
    for c in 0..N {
        rate_E[c] = frw.E[c] + dE[c];
    }
    let mut rate_U = vec![0.0; N + 1];
    for j in 0..N + 1 {
        rate_U[j] = frw.U[j] + dU[j];
    }
    let rate = State {
        E: rate_E,
        U: rate_U,
        W: rows.dW,
        M_e: frw.M_e + dM_e,
    };
    let deviation_rate = State {
        E: dE,
        U: dU,
        W: rows.dW,
        M_e: dM_e,
    };
    Ok(DerivsResult {
        rate,
        deviation_rate,
        derived: d,
        speeds: sp,
        F,
        delta_F,
        kernels,
    })
}

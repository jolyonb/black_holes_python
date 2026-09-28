//! The closure of the outer face (`pbh/outer.py`; paper Section 7.5, eq:num:sat, with the boundary ODE of Section 5.3).
//!
//! The interior rows stop one face short of the boundary; a closure supplies `d_xi U_N` and `F_N`, as deviations from
//! their FRW values, and `d_xi W`. The three closures of the Python are the three variants of `OuterClosure`, with the
//! parameters they carry there; the Python refuses any other closure before a stage runs.

use crate::eos::EquationOfState;

/// What an outer closure may look at (`OuterInputs`), plus `X_N ** 2` as the Python forms it.
pub struct OuterInputs {
    /// The radius of the outer face.
    pub X_N: f64,
    /// `X_N ** 2` by the C library's `pow`, as the Python's closures form it on the float `X_N`.
    pub X_N_squared: f64,
    /// The velocity of the outer face, `(d_xi X)_N`.
    pub X_xi_N: f64,
    /// The velocity at the outer face.
    pub U_N: f64,
    /// The auxiliary scalar of Section 7.5.
    pub W: f64,
    /// `U_N / X_N - 1`.
    pub delta_U_N: f64,
    /// `rho_{N-1} - 1`, half a cell inside the face.
    pub delta_rho_N_1: f64,
    /// The face density `<rho>_N`.
    pub rho_f_N: f64,
    /// `<rho>_N - 1`.
    pub delta_rho_f_N: f64,
    /// The face lapse `<ephi>_N`.
    pub ephi_f_N: f64,
    /// `<ephi>_N - 1`.
    pub delta_ephi_f_N: f64,
    /// The tilde mass at the outer face.
    pub mt_N: f64,
    /// `mt_N - 1`.
    pub delta_m_N: f64,
    /// `alpha (<ephi>_N U_N - h X_N)`, the fluid's velocity relative to the Hubble flow; `alpha <ephi>_N U_N` in flat
    /// spacetime.
    pub drift_N: f64,
    /// `(D_U U)_N - h`, the one-sided velocity gradient less its FRW value `h`.
    pub delta_DU_N: f64,
    /// The difference of mean-square radii across the outer face.
    pub dS_N: f64,
    /// The background sound speed.
    pub c_s: f64,
    /// The background coefficient `h`: 1 on FRW, 0 in flat spacetime.
    pub hubble: f64,
}

/// The three rows an outer closure returns (`OuterRows`).
pub struct OuterRows {
    /// `d_xi U_N - (d_xi X)_N`.
    pub delta_dU_N: f64,
    /// `F_N - F_FRW,N`.
    pub delta_F_N: f64,
    /// `d_xi W`.
    pub dW: f64,
}

/// The coefficients of the boundary ODE eq:lin:bcode (`boundary_ode_coefficients`).
pub struct BoundaryOde {
    /// `gamma_- = -(R^2 + 4 c_s^2) / (4 R (R + 2 c_s))`.
    pub gamma_minus: f64,
    /// `gamma_+ = -(R - 2 c_s) / (4 R)`.
    pub gamma_plus: f64,
    /// `gamma_0 = -c_s / (2 (R + 2 c_s))`.
    pub gamma_0: f64,
}

/// The discrete characteristic pair at the outer face, eq:lin:charvars (`characteristic_pair`).
pub struct CharacteristicPair {
    /// `u_+ = delta_U,N + kappa delta_rho,N-1`, carried outward.
    pub u_plus: f64,
    /// `u_- = delta_U,N - kappa delta_rho,N-1`, carried inward.
    pub u_minus: f64,
}

/// The outer closures of `pbh/outer.py`, each with the parameters it carries there.
pub enum OuterClosure {
    /// The outer face held at the background, `U_N = h X_N` (a test closure).
    HeldAtFrw,
    /// The exact outgoing-wave condition as a penalty, eq:num:sat, with its strengths `(tau_u, tau_rho, tau_W)`.
    OutgoingWave { tau_u: f64, tau_rho: f64, tau_W: f64 },
    /// The outer face held on a steady exterior, with its density and lapse.
    HeldExterior { rho_N: f64, ephi_N: f64 },
}

/// The base flux of eq:num:energy at a face less the FRW flux (`base_flux_deviation`):
/// `X^2 [(alpha w h X - d_xi X) (<rho> - 1) + (1 + w) drift <rho>]`, the square formed as `X * X`.
pub fn base_flux_deviation(
    X: f64,
    X_xi: f64,
    drift: f64,
    rho_f: f64,
    delta_rho_f: f64,
    eos: &EquationOfState,
    hubble: f64,
) -> f64 {
    let alpha = eos.alpha_float;
    let w = eos.w_float;
    X * X * ((alpha * w * hubble * X - X_xi) * delta_rho_f + (1.0 + w) * drift * rho_f)
}

/// The coefficients `(gamma_-, gamma_+, gamma_0)` of the boundary ODE eq:lin:bcode at radius `R`.
pub fn boundary_ode_coefficients(c_s: f64, R: f64) -> BoundaryOde {
    BoundaryOde {
        gamma_minus: -(R * R + 4.0 * c_s * c_s) / (4.0 * R * (R + 2.0 * c_s)),
        gamma_plus: -(R - 2.0 * c_s) / (4.0 * R),
        gamma_0: -c_s / (2.0 * (R + 2.0 * c_s)),
    }
}

/// The characteristic pair `u_pm = delta_U,N pm kappa delta_rho,N-1`, `kappa = 3 c_s / (2 X_N)` (eq:lin:charvars).
pub fn characteristic_pair(delta_U_N: f64, delta_rho_N_1: f64, X_N: f64, c_s: f64) -> CharacteristicPair {
    let kappa = 1.5 * c_s / X_N;
    CharacteristicPair {
        u_plus: delta_U_N + kappa * delta_rho_N_1,
        u_minus: delta_U_N - kappa * delta_rho_N_1,
    }
}

impl OuterClosure {
    /// The three outer rows for these face-`N` quantities, or the Python's `ValueError` message.
    pub fn rows(&self, inputs: &OuterInputs, eos: &EquationOfState) -> Result<OuterRows, String> {
        match self {
            OuterClosure::HeldAtFrw => Ok(held_at_frw_rows(inputs, eos)),
            OuterClosure::OutgoingWave { tau_u, tau_rho, tau_W } => {
                outgoing_wave_rows(*tau_u, *tau_rho, *tau_W, inputs, eos)
            }
            OuterClosure::HeldExterior { rho_N, ephi_N } => held_exterior_rows(*rho_N, *ephi_N, inputs, eos),
        }
    }
}

/// `HeldAtFrw.rows`: `d_xi U_N = h (d_xi X)_N`, `F_N = (cE_N - (d_xi X)_N) X_N^2 <rho>_N`, `d_xi W = 0`, as deviations.
fn held_at_frw_rows(i: &OuterInputs, eos: &EquationOfState) -> OuterRows {
    let delta_F_N = base_flux_deviation(i.X_N, i.X_xi_N, i.drift_N, i.rho_f_N, i.delta_rho_f_N, eos, i.hubble);
    OuterRows {
        delta_dU_N: 0.0,
        delta_F_N,
        dW: 0.0,
    }
}

/// `OutgoingWave.rows`, eq:num:sat: with `pen = u_- - W`,
///
/// ```text
///     d_xi U_N = (1 - alpha) U_N - (alpha / 2) <ephi>_N X_N (mt_N + 3 w <rho>_N) - Theta_N (D_U U)_N
///                - tau_u c_s (X_N^2 / dS_N) pen,
///     F_N      = the base flux at the drift of U*_N = U_N - (tau_rho X_N / 2) pen,
///     d_xi W   = gamma_- W + gamma_+ (u_+ - tau_W pen) + gamma_0 delta_m,N   (radiation; 0 otherwise),
/// ```
///
/// the first two as deviations. Refuses a moving outer face and flat spacetime, with the Python's messages.
fn outgoing_wave_rows(
    tau_u: f64,
    tau_rho: f64,
    tau_W: f64,
    inputs: &OuterInputs,
    eos: &EquationOfState,
) -> Result<OuterRows, String> {
    if inputs.X_xi_N != 0.0 {
        return Err("the outgoing-wave closure needs a static outer face, (d_xi X)_N = 0".to_string());
    }
    if inputs.hubble != 1.0 {
        return Err("the outgoing-wave closure is derived about FRW; it has no flat-spacetime form".to_string());
    }
    let alpha = eos.alpha_float;
    let w = eos.w_float;
    let X_N = inputs.X_N;
    let c_s = inputs.c_s;

    let pair = characteristic_pair(inputs.delta_U_N, inputs.delta_rho_N_1, X_N, c_s);
    let pen = pair.u_minus - inputs.W;

    let i = inputs;
    let expansion = (1.0 - alpha) * X_N * i.delta_U_N; // (1 - alpha) delta U_N
    let lapse_mass = i.delta_ephi_f_N * (i.mt_N + 3.0 * w * i.rho_f_N) + i.delta_m_N + 3.0 * w * i.delta_rho_f_N;
    let gravity = -0.5 * alpha * X_N * lapse_mass; // less its FRW value, which cancels the FRW expansion
    let advection = -i.drift_N * (1.0 + i.delta_DU_N); // Theta_N (D_U U)_N, with Theta_N = drift_N on a static face
    let penalty = -tau_u * c_s * i.X_N_squared / i.dS_N * pen;
    let delta_dU_N = expansion + gravity + advection + penalty;

    let drift_star = i.drift_N - alpha * i.ephi_f_N * 0.5 * tau_rho * X_N * pen; // the drift of U*_N
    let delta_F_N = base_flux_deviation(X_N, 0.0, drift_star, i.rho_f_N, i.delta_rho_f_N, eos, i.hubble);

    let dW = if eos.is_radiation {
        let ode = boundary_ode_coefficients(c_s, X_N);
        ode.gamma_minus * inputs.W + ode.gamma_plus * (pair.u_plus - tau_W * pen) + ode.gamma_0 * inputs.delta_m_N
    } else {
        0.0 // W = 0 for every other equation of state
    };
    Ok(OuterRows {
        delta_dU_N,
        delta_F_N,
        dW,
    })
}

/// `HeldExterior.rows`: `d_xi U_N = (1 - alpha) U_N`, `F_N = (cE_N - (d_xi X)_N) X_N^2 rho_N` with the held face
/// values, the FRW values simply subtracted. Refuses flat spacetime, with the Python's message.
fn held_exterior_rows(
    rho_N: f64,
    ephi_N: f64,
    inputs: &OuterInputs,
    eos: &EquationOfState,
) -> Result<OuterRows, String> {
    if inputs.hubble != 1.0 {
        return Err("the held exterior is a steady flow about a hole; it has no flat-spacetime form".to_string());
    }
    let alpha = eos.alpha_float;
    let w = eos.w_float;
    let X_N = inputs.X_N;
    let X_xi_N = inputs.X_xi_N;
    let cE_N = alpha * ((1.0 + w) * ephi_N * inputs.U_N - X_N);
    let F_N = (cE_N - X_xi_N) * inputs.X_N_squared * rho_N;
    let F_frw = (alpha * w * X_N - X_xi_N) * inputs.X_N_squared;
    Ok(OuterRows {
        delta_dU_N: (1.0 - alpha) * inputs.U_N - X_xi_N,
        delta_F_N: F_N - F_frw,
        dW: 0.0,
    })
}

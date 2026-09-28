//! The shock-capturing kernels (`pbh/kernels.py`; paper Section 7.7: eq:num:recon, eq:num:theta, eq:num:hll,
//! eq:num:jump, eq:num:qvisc).
//!
//! They replace two things in the centred base scheme, the energy flux `F_j` and the artificial pressure force `Q_j`:
//!
//! (a) the density's deviation reconstructed to both sides of every face, piecewise linear in `s = X^2`, limited by
//!     mc (or minmod) and scaled by the theta-limiter (`reconstruct_density`);
//! (b) the HLL flux of the one-sided fluxes with the chord-widened bounds, as its deviation from the FRW flux
//!     (`hll_flux`);
//! (c) the viscous pressure from the minmod-limited jump of the peculiar velocity across each cell, and its areal
//!     force at the faces (`viscous_pressure`), carried into the flux on each side (`viscous_sides`).
//!
//! The letters are those of `kernels.py` and of the paper's Section 7.7. The code runs (a), then (c), then (b), since
//! the flux carries the viscous pressure.
//!
//! At an excision face the kernels' own one-sided rows apply, as in the Python.

use crate::derived::Derived;
use crate::eos::EquationOfState;
use crate::geometry::Geometry;
use crate::layout::Layout;
use crate::numpy_like::{maximum, minimum, minmod2, minmod3};
use crate::state::State;
use crate::stencils::{StencilWeights, theta_limited_faces};

/// Whether the shock-capturing kernels are on (production) or the centred base scheme runs (`Kernels`).
pub enum Kernels {
    /// The three kernels of Section 7.7.
    Production,
    /// The centred base flux and no artificial pressure: a test switch.
    Centred,
}

/// The limiter of the density reconstruction (`DensityLimiter`).
pub enum DensityLimiter {
    /// Monotonized central.
    Mc,
    /// Minmod.
    Minmod,
}

/// How the viscous work enters the energy flux (`ViscousFlux`).
pub enum ViscousFlux {
    /// Both one-sided fluxes carry the face average `<q>_j`.
    Averaged,
    /// Each one-sided flux carries its own cell's `q / rho` at that side's reconstructed density (production).
    DensityWeighted,
}

/// The kernel switches of Table tab:num:params (`KernelSettings`).
pub struct KernelSettings {
    /// Production kernels, or the centred base scheme.
    pub kernels: Kernels,
    /// The limiter of the density reconstruction.
    pub density_limiter: DensityLimiter,
    /// The one constant of the viscous pressure, `1`.
    pub c_v: f64,
    /// The theta-limiter's fraction; it also fixes the outer face's density (Section 7.5).
    pub theta: f64,
    /// The viscous work term of the energy flux.
    pub viscous_flux: ViscousFlux,
    /// Whether the viscous tension is capped at the fluid pressure, `q >= -w rho`.
    pub cap_tension: bool,
}

/// What the kernels produce at one evaluation (`KernelResult`).
pub struct KernelResult {
    /// The density reconstructed to face `j` from the cell inside it.
    pub rho_L: Vec<f64>,
    /// The density reconstructed to face `j` from the cell outside it.
    pub rho_R: Vec<f64>,
    /// `rho_L - 1`.
    pub delta_rho_L: Vec<f64>,
    /// `rho_R - 1`.
    pub delta_rho_R: Vec<f64>,
    /// The limited jump of the peculiar velocity across each cell (cells).
    pub J: Vec<f64>,
    /// The artificial viscous pressure on the cells.
    pub q: Vec<f64>,
    /// Its face value `<q>_j`.
    pub q_f: Vec<f64>,
    /// Its force at the faces.
    pub Q: Vec<f64>,
    /// The HLL energy flux through the retained faces `j < N` (NaN at face `N`, the closure's).
    pub F: Vec<f64>,
    /// The theta-limiter's factor on each retained cell's slope (cells).
    pub theta_scale: Vec<f64>,
    /// The upper bound `Lambda^+_j` of the HLL flux.
    pub Lam_plus: Vec<f64>,
    /// Its lower bound `Lambda^-_j`.
    pub Lam_minus: Vec<f64>,
    /// The chord speed of the side inside each face.
    pub v_L: Vec<f64>,
    /// The chord speed of the side outside it.
    pub v_R: Vec<f64>,
}

/// The reconstructed density at both sides of every face (`reconstruct_density`'s five arrays).
pub struct Reconstruction {
    /// `1 + delta_rho_L` (faces).
    pub rho_L: Vec<f64>,
    /// `1 + delta_rho_R` (faces).
    pub rho_R: Vec<f64>,
    /// The deviation from the cell inside each face (faces).
    pub delta_rho_L: Vec<f64>,
    /// The deviation from the cell outside each face (faces).
    pub delta_rho_R: Vec<f64>,
    /// The theta-limiter's factor on each retained cell (cells).
    pub t: Vec<f64>,
}

/// The velocity jump and the viscous pressure (`viscous_pressure`'s four arrays).
pub struct ViscousPressure {
    /// The limited jump across each cell (cells).
    pub J: Vec<f64>,
    /// The viscous pressure (cells).
    pub q: Vec<f64>,
    /// Its face value (faces).
    pub q_f: Vec<f64>,
    /// Its areal force (faces).
    pub Q: Vec<f64>,
}

/// The viscous pressure each one-sided flux carries at the faces (`viscous_sides`).
pub struct ViscousSides {
    /// The inside flux's.
    pub q_L: Vec<f64>,
    /// The outside flux's.
    pub q_R: Vec<f64>,
}

/// The HLL flux and its bounds (`hll_flux`'s five arrays).
pub struct HllFlux {
    /// `F_j - F_FRW,j` at the retained faces `j < N`.
    pub delta_F: Vec<f64>,
    /// `Lambda^+_j`.
    pub Lam_plus: Vec<f64>,
    /// `Lambda^-_j`.
    pub Lam_minus: Vec<f64>,
    /// The chord speed of the side inside each face (`hll_flux`'s `chord_L`, which `calc_derivs` calls `v_L`).
    pub v_L: Vec<f64>,
    /// The chord speed of the side outside it (`chord_R`, which `calc_derivs` calls `v_R`).
    pub v_R: Vec<f64>,
}

/// One side's flux less the FRW flux, and its chord speed (`hll_flux.one_sided`).
struct OneSided {
    /// The one-sided flux less the FRW flux, `F_j(rho, q) - F_FRW,j` (the Python's `G_L` or `G_R`).
    G: f64,
    /// The chord speed `F_j(rho, q) / (X_j^2 rho)` (the Python's `v_L` or `v_R`).
    chord: f64,
}

/// The one-sided flux of eq:num:hll at one face, less the FRW flux, and its chord speed (`hll_flux.one_sided`, one
/// entry of it): `X^2 [frw_speed (rho - 1) + (1 + w) drift rho + drift q]` with
/// `drift = alpha (h X (e^phi - 1) + e^phi dU)` at the lapse of this side's density.
///
/// `X`, `X2`, `frw_speed` and `dU` are face `j`'s; `rho`, `delta_rho` and `q` are this side's. `dU = U - h X` is the
/// absolute deviation of the velocity, `deviation.U[j]`, as the Python's `dU`; not `Derived::delta_U`, which is the
/// relative deviation `U / X - 1`.
fn one_sided(
    X: f64,
    X2: f64,
    frw_speed: f64,
    dU: f64,
    rho: f64,
    delta_rho: f64,
    q: f64,
    eos: &EquationOfState,
    hubble: f64,
) -> OneSided {
    let alpha = eos.alpha_float;
    let w_eos = eos.w_float;
    let lapse = eos.lapse_and_deviation(rho, delta_rho);
    let drift = alpha * (hubble * X * lapse.delta_ephi + lapse.ephi * dU); // alpha (e^phi U - h X)
    let chord = frw_speed + (1.0 + w_eos) * drift + drift * q / rho;
    OneSided {
        G: X2 * (frw_speed * delta_rho + (1.0 + w_eos) * drift * rho + drift * q),
        chord,
    }
}

/// The density to both sides of every retained face, piecewise linear in `s = X^2` (eq:num:recon, eq:num:theta),
/// carried out on the deviation `rho - 1` with one added back.
#[inline(never)] // kept out of line: see `calc_derivs`
pub fn reconstruct_density(
    delta_rho: &[f64],
    geo: &Geometry,
    w: &StencilWeights,
    limiter: &DensityLimiter,
    theta: f64,
) -> Reconstruction {
    let N = w.layout.N;
    let j_e = w.layout.j_e;
    let dS = &geo.dS;
    // The one-sided slopes d_j across the interior retained faces j_e+1 .. N-1, indexed by face.
    let mut d = vec![f64::NAN; N + 1];
    for j in j_e + 1..N {
        d[j] = (delta_rho[j] - delta_rho[j - 1]) / dS[j];
    }
    // The limited slope of every retained cell: the interior cells from their two faces, the first and last retained
    // cells from their single adjacent difference.
    let mut slope = vec![f64::NAN; N];
    for c in j_e + 1..N - 1 {
        let d_in = d[c];
        let d_out = d[c + 1];
        slope[c] = match limiter {
            DensityLimiter::Mc => minmod3(0.5 * (d_in + d_out), w.r_L[c] * d_in, w.r_R[c] * d_out),
            DensityLimiter::Minmod => minmod2(d_in, d_out),
        };
    }
    slope[j_e] = d[j_e + 1];
    slope[N - 1] = d[N - 1];
    // The theta-limiter scales every retained cell's slope, where it must, so that both its face values are at least
    // theta times its density.
    let mut t = vec![f64::NAN; N];
    let mut delta_L = vec![f64::NAN; N + 1];
    let mut delta_R = vec![f64::NAN; N + 1];
    for c in j_e..N {
        let faces = theta_limited_faces(
            delta_rho[c],
            slope[c] * geo.s_in[c], // the slope times X^2 - sbar_c at each face
            slope[c] * geo.s_out[c],
            theta,
        );
        t[c] = faces.t;
        delta_L[c + 1] = faces.delta_out; // cell c is inside face c + 1 ...
        delta_R[c] = faces.delta_in; // ... and outside face c
    }
    delta_L[j_e] = delta_R[j_e]; // nothing inside the innermost face: transmissive
    delta_R[N] = delta_L[N];
    let mut rho_L = vec![0.0; N + 1];
    let mut rho_R = vec![0.0; N + 1];
    for j in 0..N + 1 {
        rho_L[j] = 1.0 + delta_L[j];
        rho_R[j] = 1.0 + delta_R[j];
    }
    Reconstruction {
        rho_L,
        rho_R,
        delta_rho_L: delta_L,
        delta_rho_R: delta_R,
        t,
    }
}

/// The limited velocity jump, the viscous pressure, its face value and its areal force (eq:num:jump, eq:num:qvisc).
///
/// `X_je_squared` is `X[j_e] ** 2` as the Python forms it on the numpy scalar (the C library's `pow`), for the end
/// row at an excision face.
#[inline(never)] // kept out of line: see `calc_derivs`
pub fn viscous_pressure(
    geo: &Geometry,
    d: &Derived,
    Lam: &[f64],
    eos: &EquationOfState,
    w: &StencilWeights,
    c_v: f64,
    cap_tension: bool,
    X_je_squared: f64,
) -> ViscousPressure {
    let N = w.layout.N;
    let j_e = w.layout.j_e;
    let X = &geo.X;
    let Xm = &geo.Xm;
    let dX = &geo.dX;
    let alpha = eos.alpha_float;
    let w_eos = eos.w_float;
    // The peculiar velocity, its slope in each cell, and the minmod-limited slope at each face: the single adjacent
    // difference at the innermost retained face and at the outer face.
    let mut upsilon = vec![0.0; N + 1];
    for j in 0..N + 1 {
        upsilon[j] = X[j] * d.delta_U[j]; // U - h X, from the deviation
    }
    if j_e == 0 {
        upsilon[0] = 0.0; // U_0 = X_0 = 0
    }
    let mut g = vec![f64::NAN; N];
    for c in j_e..N {
        g[c] = (upsilon[c + 1] - upsilon[c]) / dX[c];
    }
    let mut g_f = vec![f64::NAN; N + 1];
    for j in j_e + 1..N {
        g_f[j] = minmod2(g[j - 1], g[j]);
    }
    g_f[j_e] = g[j_e];
    g_f[N] = g[N - 1];
    // The limited jump across each cell: the profiles from its two faces, evaluated at the midpoint.
    let mut J = vec![f64::NAN; N];
    for c in j_e..N {
        J[c] = (upsilon[c + 1] + g_f[c + 1] * (Xm[c] - X[c + 1])) - (upsilon[c] + g_f[c] * (Xm[c] - X[c]));
    }
    // The viscous pressure on the cells, normalised by the signal speed and the inertia factor, tapered at the edge.
    let mut q = vec![f64::NAN; N];
    for c in j_e..N {
        let Lam_hat = maximum(Lam[c], Lam[c + 1]);
        let Gammabar2_hat = 0.5 * (d.Gammabar2[c] + d.Gammabar2[c + 1]);
        q[c] = -0.5 * c_v * Lam_hat * (1.0 + w_eos) * d.rho[c] / (alpha * d.ephi[c] * Gammabar2_hat) * J[c];
    }
    if cap_tension {
        // The tension capped at the fluid pressure, so that the total pressure stays non-negative (kernels.py).
        for c in j_e..N {
            q[c] = maximum(q[c], -w_eos * d.rho[c]);
        }
    }
    q[N - 1] = 0.0;
    // Its face value (the stencil's, which is the cell behind an excision face) and its areal force: the interior
    // stencil, and the kernels' own end row at the excision face.
    let q_f = w.face_average(&q);
    let mut sbar_q = vec![0.0; N];
    for c in 0..N {
        sbar_q[c] = geo.sbar[c] * q[c];
    }
    let force = w.gradient_s(&sbar_q);
    let mut Q = vec![f64::NAN; N + 1];
    for j in j_e + 1..N {
        Q[j] = force[j] / geo.X2[j];
    }
    if j_e > 0 {
        Q[j_e] = 2.0 * geo.sbar[j_e] * q[j_e] / (X_je_squared * dX[j_e]);
    } else {
        Q[0] = 0.0; // face 0 has no velocity equation
    }
    ViscousPressure { J, q, q_f, Q }
}

/// The viscous pressure each one-sided flux of eq:num:hll carries at the faces (`viscous_sides`): the face average
/// to both (`Averaged`), or each side its own cell's `q / rho` at its reconstructed density (`DensityWeighted`).
#[inline(never)] // kept out of line: see `calc_derivs`
pub fn viscous_sides(
    q: &[f64],
    q_f: &[f64],
    rho: &[f64],
    rho_L: &[f64],
    rho_R: &[f64],
    layout: &Layout,
    mode: &ViscousFlux,
) -> ViscousSides {
    let mut q_L = q_f.to_vec();
    let mut q_R = q_f.to_vec();
    if let ViscousFlux::Averaged = mode {
        return ViscousSides { q_L, q_R };
    }
    let N = layout.N;
    let j_e = layout.j_e;
    for j in j_e + 1..N {
        q_L[j] = q[j - 1] / rho[j - 1] * rho_L[j];
        q_R[j] = q[j] / rho[j] * rho_R[j];
    }
    if j_e > 0 {
        // the excision face: both sides are the first retained cell, at its reconstructed face density
        q_L[j_e] = q[j_e] / rho[j_e] * rho_R[j_e];
        q_R[j_e] = q_L[j_e];
    }
    ViscousSides { q_L, q_R }
}

/// The HLL energy flux through the retained faces `j < N` (eq:num:hll), as its deviation from the FRW flux.
///
/// Each side's flux less the FRW flux is `X^2 [frw_speed (rho - 1) + (1 + w) drift rho + drift q]`, with
/// `drift = alpha (h X (e^phi - 1) + e^phi dU)` at that side's lapse; the bounds are
/// `Lambda^+ = max(Theta + a, v^L, v^R, 0)` and `Lambda^- = min(Theta - a, v^L, v^R, 0)`, taken pairwise in the order
/// printed.
#[inline(never)] // kept out of line: see `calc_derivs`
pub fn hll_flux(
    rho_L: &[f64],
    rho_R: &[f64],
    delta_rho_L: &[f64],
    delta_rho_R: &[f64],
    q_L: &[f64],
    q_R: &[f64],
    deviation: &State,
    geo: &Geometry,
    Theta: &[f64],
    a: &[f64],
    eos: &EquationOfState,
    w: &StencilWeights,
    hubble: f64,
    frw_speed: &[f64],
) -> HllFlux {
    let N = w.layout.N;
    let j_e = w.layout.j_e;

    let mut Lam_plus = vec![f64::NAN; N + 1];
    let mut Lam_minus = vec![f64::NAN; N + 1];
    let mut chord_L = vec![f64::NAN; N + 1];
    let mut chord_R = vec![f64::NAN; N + 1];
    let mut delta_F = vec![f64::NAN; N + 1];
    for j in j_e..N {
        let (X, X2, speed, dU) = (geo.X[j], geo.X2[j], frw_speed[j], deviation.U[j]);
        let left = one_sided(X, X2, speed, dU, rho_L[j], delta_rho_L[j], q_L[j], eos, hubble);
        let right = one_sided(X, X2, speed, dU, rho_R[j], delta_rho_R[j], q_R[j], eos, hubble);
        let Lp = maximum(maximum(maximum(Theta[j] + a[j], left.chord), right.chord), 0.0);
        let Lm = minimum(minimum(minimum(Theta[j] - a[j], left.chord), right.chord), 0.0);
        Lam_plus[j] = Lp;
        Lam_minus[j] = Lm;
        chord_L[j] = left.chord;
        chord_R[j] = right.chord;
        delta_F[j] = (Lp * left.G - Lm * right.G + Lp * Lm * geo.X2[j] * (delta_rho_R[j] - delta_rho_L[j])) / (Lp - Lm);
    }
    if j_e == 0 {
        delta_F[0] = 0.0;
    }
    HllFlux {
        delta_F,
        Lam_plus,
        Lam_minus,
        v_L: chord_L,
        v_R: chord_R,
    }
}

//! The stencils that carry a field across a face (`pbh/stencils.py`; paper eq:num:stencils, at an excision face
//! eq:numbh:rows1).
//!
//! Around face `j`, cells `j - 1` and `j` share it:
//!
//! ```text
//!     <f>_j     = (f_{j-1} + f_j) / 2
//!     (D_s f)_j = 2 X_j (f_j - f_{j-1}) / dS_j
//!     (D_U U)_j = (U_{j+1} - U_{j-1}) / (X_{j+1} - X_{j-1})
//! ```
//!
//! At the origin `<f>_0 = f_0` and `(D_s f)_0 = 0`; at an excision face `<f>_je = f_je`, no pressure gradient, and the
//! velocity gradient the one retained difference; at the outer face the density is the last cell's theta-limited
//! reconstruction and the velocity gradient the three-point one-sided row. The weights are formed once per frame
//! (`StencilWeights::of`, the Python's `StencilWeights.of`). Outputs are face arrays, NaN where the operator is not
//! defined.

use crate::geometry::Geometry;
use crate::layout::Layout;
use crate::numpy_like::{maximum, minimum};

/// The geometry-dependent coefficients of the three stencils and of the mc limiter (`pbh.stencils.StencilWeights`).
pub struct StencilWeights {
    /// Which faces are retained and where the excision rows apply.
    pub layout: Layout,
    /// `2 X_j / dS_j` (faces `1..N-1`; NaN elsewhere).
    pub grad_s: Vec<f64>,
    /// `1 / (X_{j+1} - X_{j-1})` (faces `1..N-1`; NaN elsewhere).
    pub centred_U: Vec<f64>,
    /// The coefficients of `U_N`, `U_{N-1}`, `U_{N-2}` in the one-sided row at the outer face.
    pub outer_U: [f64; 3],
    /// `1 / dX_{j_e}`, the factor of the one retained difference at an excision face.
    pub excision_U: f64,
    /// The last cell's `(sbar_{N-1} - sbar_{N-2}, X_{N-1}^2 - sbar_{N-1}, X_N^2 - sbar_{N-1})`.
    pub outer_rho: [f64; 3],
    /// `dS_c / (sbar_c - X_c^2)`, the mc limiter's inner ratio of eq:num:recon (cells `1..N-2`).
    pub r_L: Vec<f64>,
    /// `dS_{c+1} / (X_{c+1}^2 - sbar_c)`, its outer ratio (cells `1..N-2`).
    pub r_R: Vec<f64>,
}

/// A density at the outer face and its deviation (`StencilWeights.outer_face_density`).
pub struct FaceDensity {
    /// `rho_hat_N`.
    pub rho: f64,
    /// `rho_hat_N - 1`.
    pub delta_rho: f64,
}

/// One cell's theta-limited profile (`theta_limited_faces`, one entry of it).
pub struct ThetaLimited {
    /// The scale factor `t` of the cell's slope, `1` where the limiter did not bind.
    pub t: f64,
    /// The deviation `rho - 1` of its inner face value.
    pub delta_in: f64,
    /// The deviation `rho - 1` of its outer face value.
    pub delta_out: f64,
}

impl StencilWeights {
    /// The weights for this geometry and these retained faces (`StencilWeights.of`).
    pub fn of(geo: &Geometry, layout: Layout) -> StencilWeights {
        let N = layout.N;
        let j_e = layout.j_e;
        let X = &geo.X;
        let dS = &geo.dS;
        let mut grad_s = vec![f64::NAN; N + 1];
        let mut centred_U = vec![f64::NAN; N + 1];
        for j in 1..N {
            grad_s[j] = 2.0 * X[j] / dS[j];
            centred_U[j] = 1.0 / (X[j + 1] - X[j - 1]);
        }
        let outer_U = one_sided_three_point(geo.dX[N - 1], geo.dX[N - 2]);
        let excision_U = 1.0 / geo.dX[j_e]; // the one retained difference at an excision face
        let outer_rho = [dS[N - 1], geo.s_in[N - 1], geo.s_out[N - 1]];
        // The mc ratios over the cells with two faces inside the grid: the difference quotient across each face over
        // that face's offset in `s`. The negation is taken first, as the Python takes it: `-dS / s_in`.
        let mut r_L = vec![f64::NAN; N];
        let mut r_R = vec![f64::NAN; N];
        for c in 1..N - 1 {
            r_L[c] = -dS[c] / geo.s_in[c];
            r_R[c] = dS[c + 1] / geo.s_out[c];
        }
        StencilWeights {
            layout,
            grad_s,
            centred_U,
            outer_U,
            excision_U,
            outer_rho,
            r_L,
            r_R,
        }
    }

    /// The face value `<f>_j` of a cell field (eq:num:stencils, first line; eq:numbh:rows1 at `j_e`): the two-cell
    /// average inside, `f_0` at the origin, the cell behind an excision face, NaN at face `N`.
    pub fn face_average(&self, f: &[f64]) -> Vec<f64> {
        let N = self.layout.N;
        let j_e = self.layout.j_e;
        let mut avg = vec![f64::NAN; N + 1];
        for j in j_e + 1..N {
            avg[j] = 0.5 * (f[j - 1] + f[j]);
        }
        avg[j_e] = f[j_e];
        avg
    }

    /// The density `rho_hat_N` at the outer face and its deviation (Section 7.5, eq:num:theta): the last cell's
    /// one-sided slope in `s`, theta-limited, evaluated at face `N`.
    pub fn outer_face_density(&self, delta_rho: &[f64], theta: f64) -> FaceDensity {
        let N = self.layout.N;
        let [dS, s_in, s_out] = self.outer_rho;
        let last = delta_rho[N - 1];
        let slope = (last - delta_rho[N - 2]) / dS; // formed as the reconstruction forms it, to the last bit
        let faces = theta_limited_faces(last, slope * s_in, slope * s_out, theta);
        FaceDensity {
            rho: 1.0 + faces.delta_out,
            delta_rho: faces.delta_out,
        }
    }

    /// The gradient `(D_s f)_j = 2 X_j (f_j - f_{j-1}) / dS_j` of a cell field (eq:num:stencils, second line): zero
    /// at the origin and at an excision face, NaN at face `N`.
    pub fn gradient_s(&self, f: &[f64]) -> Vec<f64> {
        let N = self.layout.N;
        let j_e = self.layout.j_e;
        let mut grad = vec![f64::NAN; N + 1];
        for j in j_e + 1..N {
            grad[j] = self.grad_s[j] * (f[j] - f[j - 1]);
        }
        grad[j_e] = 0.0; // none at the origin, and none formed at an excision face
        grad
    }

    /// The velocity gradient `(D_U U)_j` (eq:num:stencils, third and fourth lines; eq:numbh:rows1 at `j_e`): the
    /// centred two-face difference inside, the three-point one-sided row at face `N`, the one retained difference at
    /// an excision face, and NaN at the origin.
    pub fn velocity_gradient(&self, U: &[f64]) -> Vec<f64> {
        let N = self.layout.N;
        let j_e = self.layout.j_e;
        let mut grad = vec![f64::NAN; N + 1];
        for j in j_e + 1..N {
            grad[j] = self.centred_U[j] * (U[j + 1] - U[j - 1]);
        }
        let [a_0, a_1, a_2] = self.outer_U;
        grad[N] = a_0 * U[N] + a_1 * U[N - 1] + a_2 * U[N - 2];
        if j_e > 0 {
            grad[j_e] = self.excision_U * (U[j_e + 1] - U[j_e]);
        }
        grad
    }
}

/// The two face values of a cell whose linear profile is scaled by the theta-limiter (eq:num:theta).
///
/// The cell keeps its mean and has its offsets scaled by `t = min(1, (1 - theta) rho_c / m)`, `m` the larger drop
/// below the mean, so that both face values are at least `theta rho_c`. The ratio is only taken of a drop above
/// `allowed`, as the Python's `np.where` keeps it; a NaN drop leaves `t = 1`, as there.
pub fn theta_limited_faces(delta_rho: f64, off_in: f64, off_out: f64, theta: f64) -> ThetaLimited {
    let rho = 1.0 + delta_rho;
    let drop = -minimum(minimum(off_in, off_out), 0.0); // the larger drop below the mean, >= 0
    let allowed = (1.0 - theta) * rho;
    let t = if drop > allowed {
        allowed / maximum(drop, allowed)
    } else {
        1.0
    };
    ThetaLimited {
        t,
        delta_in: delta_rho + t * off_in,
        delta_out: delta_rho + t * off_out,
    }
}

/// The coefficients of the derivative at the end of three points with spacings `Delta_1` (nearest) and `Delta_2`
/// (`StencilWeights._one_sided_three_point`): the fourth line of eq:num:stencils, written for the outer face,
/// `(D_U U)_N = a_0 U_N + a_1 U_{N-1} + a_2 U_{N-2}`.
fn one_sided_three_point(Delta_1: f64, Delta_2: f64) -> [f64; 3] {
    [
        (2.0 * Delta_1 + Delta_2) / (Delta_1 * (Delta_1 + Delta_2)),
        -(Delta_1 + Delta_2) / (Delta_1 * Delta_2),
        Delta_1 / (Delta_2 * (Delta_1 + Delta_2)),
    ]
}

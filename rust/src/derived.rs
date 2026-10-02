//! The fields derived from the state at every stage (`pbh/derived.py`; paper Table tab:num:layout, Section 7.3).
//!
//! In this order: the cell density `rho_c = E_c / Delta V_c`, asserted positive; its deviation; the cell lapse and
//! its deviation; the cumulative mass `M_j = M_e + 3 sum E_i` (a sequential running sum, as `np.cumsum` forms it) and
//! its deviation; the tilde mass `mt_j = M_j / X_j^3`; the relative deviations `delta_U` and `delta_m`;
//! `Gammabar_j^2` from the deviation, asserted positive; and the face values of the two even cell fields, with the
//! outer face's one state (Section 7.5).
//!
//! The hyperbolicity checks are the Python's, in its order: every retained `rho_c` first, the first offender by index,
//! then `Gammabar_j^2` over the retained faces, then the outer face's density (as `rho` at index `N`). NaN counts as
//! not positive.
//!
//! A cell stored whole (`storage.rs`) has its content itself in `state.E` and its rounded `E - Delta V` in the
//! deviation: its `delta_rho` and lapse come from its density.

use crate::eos::{Background, EquationOfState};
use crate::geometry::Geometry;
use crate::state::State;
use crate::stencils::StencilWeights;

/// A non-positive density or `Gammabar^2` (`NotHyperbolicError`): the system has left its hyperbolic domain.
pub struct NotHyperbolic {
    /// `"rho"` (a cell, or at index `N` the outer face's density) or `"Gammabar2"` (a face).
    pub field: &'static str,
    /// The cell or face index of the first offending entry.
    pub index: usize,
    /// The offending value.
    pub value: f64,
}

/// The fields of Table tab:num:layout derived from the state at one stage (`pbh.derived.Derived`).
pub struct Derived {
    /// `rho_c = E_c / Delta V_c` (cells).
    pub rho: Vec<f64>,
    /// `ephi_c = rho_c ^ (-w / (1 + w))` (cells).
    pub ephi: Vec<f64>,
    /// `ephi_c - 1` (cells).
    pub delta_ephi: Vec<f64>,
    /// The cumulative mass `M_j` (faces).
    pub M: Vec<f64>,
    /// `M_j - X_j^3 = delta M_e + 3 sum delta E_i` (faces).
    pub delta_M: Vec<f64>,
    /// `mt_j = M_j / X_j^3` (faces `1..N`; NaN at the origin).
    pub mt: Vec<f64>,
    /// `rho_c - 1 = delta E_c / Delta V_c` (cells).
    pub delta_rho: Vec<f64>,
    /// `U_j / X_j - 1 = delta U_j / X_j` (faces `1..N`).
    pub delta_U: Vec<f64>,
    /// `mt_j - 1 = delta M_j / X_j^3` (faces `1..N`).
    pub delta_m: Vec<f64>,
    /// `Gammabar_j^2` (faces), the first line of eq:num:facefields.
    pub Gammabar2: Vec<f64>,
    /// `<rho>_j` (faces), with the outer face's theta-limited value at `N`.
    pub rho_f: Vec<f64>,
    /// `<ephi>_j` (faces), with the lapse of `rho_f[N]` at `N`.
    pub ephi_f: Vec<f64>,
    /// `<rho>_j - 1` (faces).
    pub delta_rho_f: Vec<f64>,
    /// `<ephi>_j - 1` (faces).
    pub delta_ephi_f: Vec<f64>,
}

/// Form the derived fields of one stage, asserting hyperbolicity on the retained entries (`derive`).
///
/// `deviation` is the state's deviation from FRW, which `Gammabar^2` and the relative deviations read. `whole` marks
/// the cells stored whole (`storage.rs`), `None` if none is: their `delta_rho` is `rho - 1`, formed from their density,
/// and their lapse is the power of their density (`EquationOfState::lapse_whole`). `ANY` is whether `whole` is `Some`
/// (`calc_derivs`).
#[inline(never)] // kept out of line: see `calc_derivs`
pub fn derive<const ANY: bool>(
    state: &State,
    geo: &Geometry,
    bg: &Background,
    eos: &EquationOfState,
    w: &StencilWeights,
    theta: f64,
    deviation: &State,
    whole: Option<&[bool]>,
) -> Result<Derived, NotHyperbolic> {
    let whole = whole.filter(|_| ANY); // `None` in the instance for none stored whole, where the tests fold away
    let N = w.layout.N;
    let j_e = w.layout.j_e;

    let mut rho = vec![f64::NAN; N];
    for c in j_e..N {
        rho[c] = state.E[c] / geo.dV[c];
    }
    assert_positive(&rho, j_e, "rho")?;
    let mut delta_rho = vec![f64::NAN; N];
    for c in j_e..N {
        delta_rho[c] = deviation.E[c] / geo.dV[c];
    }
    if let Some(whole) = whole {
        for c in 0..N {
            if whole[c] {
                delta_rho[c] = rho[c] - 1.0; // a cell stored whole: from its own density, not its rounded deviation
            }
        }
    }

    let mut ephi = vec![f64::NAN; N];
    let mut delta_ephi = vec![f64::NAN; N];
    for c in j_e..N {
        let lapse = match whole {
            Some(whole) if whole[c] => eos.lapse_whole(rho[c]),
            _ => eos.lapse_and_deviation(rho[c], delta_rho[c]),
        };
        ephi[c] = lapse.ephi;
        delta_ephi[c] = lapse.delta_ephi;
    }

    // The mass inside each retained face: what is inside the innermost retained face (nothing, or the excised M_e)
    // plus three times the energy of every retained cell inside it (Sections 7.2 and 8.3).
    // `np.cumsum` is the running sum from the first retained cell, which is where it starts (not from 0.0 + E).
    let mut M = vec![f64::NAN; N + 1];
    M[j_e] = state.M_e;
    let mut sum = state.E[j_e];
    M[j_e + 1] = state.M_e + 3.0 * sum;
    for c in j_e + 1..N {
        sum += state.E[c];
        M[c + 1] = state.M_e + 3.0 * sum;
    }

    let mut mt = vec![f64::NAN; N + 1];
    let inner = j_e.max(1); // never face 0, whose mt is 0 / 0
    for j in inner..N + 1 {
        mt[j] = M[j] / geo.X3[j];
    }

    let mut dM = vec![f64::NAN; N + 1]; // M - X^3, the mass against its FRW value, as a sum of small terms
    dM[j_e] = deviation.M_e;
    let mut delta_sum = deviation.E[j_e];
    dM[j_e + 1] = deviation.M_e + 3.0 * delta_sum;
    for c in j_e + 1..N {
        delta_sum += deviation.E[c];
        dM[c + 1] = deviation.M_e + 3.0 * delta_sum;
    }

    let mut delta_U = vec![f64::NAN; N + 1];
    for j in inner..N + 1 {
        delta_U[j] = deviation.U[j] / geo.X[j];
    }
    let mut delta_m = vec![f64::NAN; N + 1];
    for j in inner..N + 1 {
        delta_m[j] = dM[j] / geo.X3[j];
    }

    let mut Gammabar2 = vec![f64::NAN; N + 1];
    for j in inner..N + 1 {
        Gammabar2[j] = gammabar_squared(bg, geo.X[j], state.U[j], deviation.U[j], dM[j]);
    }
    if j_e == 0 {
        Gammabar2[0] = bg.Gammabar2; // M / X ~ X^2 -> 0 at the origin, and U_0 = 0
    }
    assert_positive(&Gammabar2, j_e, "Gammabar2")?;

    // The face values: the two-cell averages, and at the outer face its one state, the last cell's theta-limited
    // density there and the lapse of that same density (Section 7.5), never an extrapolation.
    let mut rho_f = w.face_average(&rho);
    let mut delta_rho_f = w.face_average(&delta_rho);
    let mut ephi_f = w.face_average(&ephi);
    let mut delta_ephi_f = w.face_average(&delta_ephi);
    let outer = w.outer_face_density(&delta_rho, theta);
    rho_f[N] = outer.rho;
    delta_rho_f[N] = outer.delta_rho;
    // At least theta rho_{N-1} > 0 in exact arithmetic, but `1 + delta` rounds to zero once rho_{N-1} is below ~1e-16
    assert_positive(&rho_f, N, "rho")?;
    let lapse_N = eos.lapse_and_deviation(rho_f[N], delta_rho_f[N]);
    ephi_f[N] = lapse_N.ephi;
    delta_ephi_f[N] = lapse_N.delta_ephi;

    Ok(Derived {
        rho,
        ephi,
        delta_ephi,
        M,
        delta_M: dM,
        mt,
        delta_rho,
        delta_U,
        delta_m,
        Gammabar2,
        rho_f,
        ephi_f,
        delta_rho_f,
        delta_ephi_f,
    })
}

/// `Gammabar^2 = Gammabar_FRW^2 + delta U (U + h X) - h delta M / X` at a face `X > 0`, without the FRW cancellation
/// (`gammabar_squared`, one entry of it).
pub fn gammabar_squared(bg: &Background, X: f64, U: f64, dU: f64, dM: f64) -> f64 {
    let h = bg.hubble;
    bg.Gammabar2 + dU * (U + h * X) - h * dM / X
}

/// `NotHyperbolic` at the first retained entry, from `first` on, that is not positive (NaN counts as not positive).
fn assert_positive(values: &[f64], first: usize, name: &'static str) -> Result<(), NotHyperbolic> {
    for index in first..values.len() {
        if values[index] > 0.0 {
            continue;
        }
        return Err(NotHyperbolic {
            field: name,
            index,
            value: values[index],
        });
    }
    Ok(())
}

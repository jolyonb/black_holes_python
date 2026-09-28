//! The evolved unknowns at one time, and the FRW reference on one geometry (`pbh/state.py`; paper Section 7.6).

use crate::geometry::Geometry;
use crate::numpy_like::c_pow;

/// The evolved unknowns, or their rates, at one time (`pbh.state.State`).
pub struct State {
    /// The cell energies `E_c` (cells), NaN below the excision face.
    pub E: Vec<f64>,
    /// The face velocities `U_j` (faces); `U_0 = 0` while the origin is in the domain.
    pub U: Vec<f64>,
    /// The incoming amplitude at the outer face (Section 7.5).
    pub W: f64,
    /// The mass inside the excision face; `0` while there is no excision.
    pub M_e: f64,
}

impl State {
    /// The entrywise sum `self + other`, every field (`State.plus`). `W` is the sum too, so a deviation's `-0.0`
    /// comes out as `+0.0` exactly as in the Python.
    pub fn plus(&self, other: &State) -> State {
        let mut E = vec![0.0; self.E.len()];
        for c in 0..self.E.len() {
            E[c] = self.E[c] + other.E[c];
        }
        let mut U = vec![0.0; self.U.len()];
        for j in 0..self.U.len() {
            U[j] = self.U[j] + other.U[j];
        }
        State {
            E,
            U,
            W: self.W + other.W,
            M_e: self.M_e + other.M_e,
        }
    }
}

/// The background on one geometry, as a stage uses it (`pbh.state.FrwReference`), formed once per frame.
pub struct FrwReference {
    /// The background state `frw_state(geo, j_e, h)`: `E = dV`, `U = h X`, `W = 0`, `M_e = X_je ** 3` (by `pow`).
    pub state: State,
    /// Its time derivative `frw_rate(geo, j_e, h)`.
    pub rate: State,
    /// `alpha w h X_j - (d_xi X)_j` (faces): the FRW flux is `frw_speed X^2`.
    pub frw_speed: Vec<f64>,
    /// The FRW energy flux `frw_speed X_j^2` (faces).
    pub F_frw: Vec<f64>,
}

impl FrwReference {
    /// The reference on this geometry (`FrwReference.of`), for `alpha w h` the product `alpha * w * h` of the Python's
    /// floats in its order, and `h` the background coefficient of `Background`.
    ///
    /// `frw_state` and `frw_rate` take `X_{j_e}^3` and `X_{j_e}^2` by the scalar power, the C library's `pow`, not the
    /// products in `geo.X3` and `geo.X2`, which differ from it in the last bit (see `Geometry`).
    pub fn of(geo: &Geometry, alpha_w_h: f64, j_e: usize, hubble: f64) -> FrwReference {
        let faces = geo.X.len();
        let mut U = vec![0.0; faces];
        let mut U_rate = vec![0.0; faces];
        let mut frw_speed = vec![0.0; faces];
        let mut F_frw = vec![0.0; faces];
        for j in 0..faces {
            U[j] = hubble * geo.X[j];
            U_rate[j] = hubble * geo.X_xi[j];
            frw_speed[j] = alpha_w_h * geo.X[j] - geo.X_xi[j];
            F_frw[j] = frw_speed[j] * geo.X2[j];
        }
        let X_e = geo.X[j_e];
        let X_xi_e = geo.X_xi[j_e];
        FrwReference {
            state: State {
                E: geo.dV.clone(),
                U,
                W: 0.0,
                M_e: c_pow(X_e, 3.0),
            },
            rate: State {
                E: geo.dV_xi.clone(),
                U: U_rate,
                W: 0.0,
                M_e: 3.0 * c_pow(X_e, 2.0) * X_xi_e,
            },
            frw_speed,
            F_frw,
        }
    }
}

/// The state `y_FRW + delta y` from the unpacked deviation (`Scheme.whole_state`): `reference.state.plus(deviation)`.
pub fn whole_state(reference: &FrwReference, deviation: &State) -> State {
    reference.state.plus(deviation)
}

/// The deviation `state - y_FRW` recovered from a whole state (`deviation_from_frw`), with `W` copied.
///
/// The Python forms `frw_state(geo, j_e, h)` afresh; the reference's state is that same object's values.
pub fn deviation_from_frw(state: &State, reference: &FrwReference) -> State {
    let frw = &reference.state;
    let mut E = vec![0.0; state.E.len()];
    for c in 0..state.E.len() {
        E[c] = state.E[c] - frw.E[c];
    }
    let mut U = vec![0.0; state.U.len()];
    for j in 0..state.U.len() {
        U[j] = state.U[j] - frw.U[j];
    }
    State {
        E,
        U,
        W: state.W,
        M_e: state.M_e - frw.M_e,
    }
}

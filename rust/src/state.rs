//! The evolved unknowns at one time, and the FRW reference on one geometry (`pbh/state.py`; paper Section 7.6).

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

/// The background on one geometry, as a stage uses it (`pbh.state.FrwReference`), copied from the Python's.
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

//! How each cell's energy is stored: as its deviation from FRW, or whole (`pbh/storage.py`; paper Section 7.6).
//!
//! The flags `whole` mark the cells that store their content `E_c` itself rather than its deviation `E_c - Delta V_c`;
//! the stage receives them as `Some` only when at least one cell is stored whole (`storage.any_whole`), and with `None`
//! everything is the deviation form, to the bit. Which cells move across the storage lines, and when, is the
//! Python's (`storage.switched`, at a step boundary); the stage only reads the flags.

use crate::state::State;

/// The faces with a cell stored whole on either side, `N + 1` of them for `N` cells (`faces_beside`).
pub fn faces_beside(whole: &[bool]) -> Vec<bool> {
    let mut beside = vec![false; whole.len() + 1];
    for c in 0..whole.len() {
        if whole[c] {
            beside[c] = true;
            beside[c + 1] = true;
        }
    }
    beside
}

/// The state the stored numbers stand for (`whole_of`): FRW plus the deviation, and the content itself in the cells
/// stored whole. With `None` this is `Scheme.whole_state`, `reference.state.plus(deviation)`.
#[inline(never)] // kept out of line: see `calc_derivs`
pub fn whole_of(stored: &State, whole: Option<&[bool]>, frw: &State) -> State {
    let mut state = frw.plus(stored);
    if let Some(whole) = whole {
        for c in 0..whole.len() {
            if whole[c] {
                state.E[c] = stored.E[c];
            }
        }
    }
    state
}

/// The deviation from FRW of every entry (`deviation_of`): the stored numbers, and `E - Delta V` in the cells stored
/// whole (rounded, as any deviation is). `dV` is the FRW content of each cell.
#[inline(never)] // kept out of line: see `calc_derivs`
pub fn deviation_of(stored: State, whole: Option<&[bool]>, dV: &[f64]) -> State {
    let mut deviation = stored;
    if let Some(whole) = whole {
        for c in 0..whole.len() {
            if whole[c] {
                deviation.E[c] -= dV[c];
            }
        }
    }
    deviation
}

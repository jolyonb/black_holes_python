//! Which entries are unknowns, and how they are packed into one vector (`pbh/layout.py`; paper Table tab:num:layout).
//!
//! The packed order is `[ E_c for retained cells | U_j for evolved faces | M_e (only if excised) | W ]`. Arrays are
//! full length, NaN below the excision face.

use std::ops::Range;

use crate::state::State;

/// The retained ranges of the grid and the packing of the unknowns (`pbh.layout.Layout`).
pub struct Layout {
    /// The number of cells.
    pub N: usize,
    /// The excision face; `0` before excision. Always `j_e <= N - 2`, with `N >= 2`, as the Python's `Layout` requires:
    /// `StageFrame::new` refuses anything else, so the subtractions `N - j_e` below cannot wrap.
    pub j_e: usize,
}

impl Layout {
    /// Whether cells have been dropped, in which case `M_e` is an unknown.
    pub fn excised(&self) -> bool {
        self.j_e > 0
    }

    /// The retained cells, `c = j_e .. N-1`.
    pub fn cells(&self) -> Range<usize> {
        self.j_e..self.N
    }

    /// The faces whose velocity is an unknown, `j = max(j_e, 1) .. N`.
    pub fn faces_evolved(&self) -> Range<usize> {
        self.j_e.max(1)..self.N + 1
    }

    /// The length of the packed vector.
    pub fn size(&self) -> usize {
        let n_cells = self.N - self.j_e;
        let n_faces = self.N + 1 - self.j_e.max(1);
        n_cells + n_faces + usize::from(self.excised()) + 1
    }

    /// The state from a packed vector (`Layout.unpack`): full-length arrays, NaN below `j_e`, and `U_0 = 0`.
    ///
    /// Refuses a vector of the wrong length with the Python's message.
    pub fn unpack(&self, y: &[f64]) -> Result<State, String> {
        if y.len() != self.size() {
            return Err(format!(
                "expected a packed vector of length {}, got shape ({},)",
                self.size(),
                y.len()
            ));
        }
        let n_cells = self.N - self.j_e;
        let n_faces = self.N + 1 - self.j_e.max(1);
        let mut E = vec![f64::NAN; self.N];
        for c in self.cells() {
            E[c] = y[c - self.j_e];
        }
        let mut U = vec![f64::NAN; self.N + 1];
        let first_face = self.j_e.max(1);
        for j in self.faces_evolved() {
            U[j] = y[n_cells + (j - first_face)];
        }
        if !self.excised() {
            U[0] = 0.0;
        }
        let M_e = if self.excised() { y[n_cells + n_faces] } else { 0.0 };
        Ok(State {
            E,
            U,
            W: y[y.len() - 1],
            M_e,
        })
    }
}

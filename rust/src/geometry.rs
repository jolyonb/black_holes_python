//! The exact geometry of the cells at one time (`pbh/geometry.py`; paper Section 7.2, eq:num:geom).
//!
//! The Python builds it (`Geometry.of`) and the Rust frame copies it once; nothing here recomputes it. `dV_xi` is not
//! copied: a stage reads the volume rate only through the FRW reference's rate. Indexing is the
//! Python's: face arrays have `N + 1` entries indexed by `j`, cell arrays `N` entries indexed by `c`, and `sbar` has
//! the virtual outer cell appended as entry `N`.

/// The map at the faces and the cell geometry built from it (`pbh.geometry.Geometry`).
pub struct Geometry {
    /// The scaled areal radius `X_j` of each face (faces); `X_0 = 0`.
    pub X: Vec<f64>,
    /// The velocity of each face, `(d_xi X)_j` (faces); zero on a static map.
    pub X_xi: Vec<f64>,
    /// The shell volume `Delta V_c` (cells), the FRW cell energy.
    pub dV: Vec<f64>,
    /// The mean-square radius `sbar_c` (cells `0..N`, entry `N` the virtual outer cell).
    pub sbar: Vec<f64>,
    /// `Delta S_j = sbar_c - sbar_(c-1)` (faces; entry 0 NaN).
    pub dS: Vec<f64>,
    /// The cell width `Delta X_c` (cells).
    pub dX: Vec<f64>,
    /// The cell midpoint `X_m,c` (cells).
    pub Xm: Vec<f64>,
    /// `X_j^2` (faces), the product `X X` as numpy forms an array's square.
    pub X2: Vec<f64>,
    /// `X_j^3` (faces), the product `X X X`.
    pub X3: Vec<f64>,
    /// `X_c^2 - sbar_c`, the offset in `s` of each cell's inner face from its mean (cells).
    pub s_in: Vec<f64>,
    /// `X_(c+1)^2 - sbar_c`, the same for its outer face (cells).
    pub s_out: Vec<f64>,
}

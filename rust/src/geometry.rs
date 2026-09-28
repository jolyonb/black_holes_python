//! The exact geometry of the cells at one time (`pbh/geometry.py`; paper Section 7.2, eq:num:geom).
//!
//! Built once per frame from the map at the faces (`Geometry::of`, the Python's `Geometry.of` with the same operations
//! in the same order), and handed back to the Python for the rest of the run to read. Indexing is the Python's: face
//! arrays have `N + 1` entries indexed by `j`, cell arrays `N` entries indexed by `c`, and `sbar` has the virtual outer
//! cell appended as entry `N`.

/// Whether the radii are a map's (`pbh.geometry.check_radii`): `X_0 = 0` exactly and every difference `X_(j+1) - X_j`
/// positive, formed as numpy forms `np.diff(X) > 0` (a NaN or an infinity fails, as there).
pub fn radii_admissible(X: &[f64]) -> bool {
    !X.is_empty() && X[0] == 0.0 && X.windows(2).all(|pair| pair[1] - pair[0] > 0.0)
}

/// The map at the faces and the cell geometry built from it (`pbh.geometry.Geometry`).
pub struct Geometry {
    /// The scaled areal radius `X_j` of each face (faces); `X_0 = 0`.
    pub X: Vec<f64>,
    /// The velocity of each face, `(d_xi X)_j` (faces); zero on a static map.
    pub X_xi: Vec<f64>,
    /// The shell volume `Delta V_c` (cells), the FRW cell energy.
    pub dV: Vec<f64>,
    /// Its rate on a moving map, `d_xi Delta V_c` (cells), the FRW rate of the cell energy.
    pub dV_xi: Vec<f64>,
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

impl Geometry {
    /// The geometry from the map at the `N + 2` faces `0..N+1`, the last the virtual face beyond the boundary
    /// (`Geometry.of`). The caller has checked what the Python checks there: `X_0 = 0` and the radii increasing.
    ///
    /// Over every cell, the virtual one included, `(X_-, X_+)` is `(X[j], X[j + 1])`. The volume is `(X_+ - X_-)`
    /// times a quadratic and the mean-square radius a quartic over that quadratic, each summed in the Python's order.
    pub fn of(X_all: &[f64], X_xi_all: &[f64]) -> Geometry {
        let N = X_all.len() - 2;
        let pairs = N + 1; // the N cells and the virtual one
        let mut dV_all = vec![0.0; pairs];
        let mut sbar = vec![0.0; pairs];
        let mut X2 = vec![0.0; pairs]; // X_-^2 over the pairs is X_j^2 at faces 0..N
        let mut X_plus2 = vec![0.0; pairs];
        for c in 0..pairs {
            let X_minus = X_all[c];
            let X_plus = X_all[c + 1];
            let X_minus2 = X_minus * X_minus;
            let X_plus_2 = X_plus * X_plus;
            let product = X_minus * X_plus;
            let quadratic = X_minus2 + product + X_plus_2;
            let quartic =
                X_minus2 * X_minus2 + X_plus_2 * X_plus_2 + product * (X_minus2 + X_plus_2) + product * product;
            // `shell_volumes`, which forms its own quadratic by the same operations
            dV_all[c] = (X_plus - X_minus) * quadratic / 3.0;
            sbar[c] = 0.6 * quartic / quadratic;
            X2[c] = X_minus2;
            X_plus2[c] = X_plus_2;
        }
        let mut dV_xi = vec![0.0; N];
        let mut dX = vec![0.0; N];
        let mut Xm = vec![0.0; N];
        let mut s_in = vec![0.0; N];
        let mut s_out = vec![0.0; N];
        for c in 0..N {
            let X_minus = X_all[c];
            let X_plus = X_all[c + 1];
            dV_xi[c] = X_plus2[c] * X_xi_all[c + 1] - X2[c] * X_xi_all[c];
            dX[c] = X_plus - X_minus;
            Xm[c] = 0.5 * (X_minus + X_plus);
            s_in[c] = X2[c] - sbar[c];
            s_out[c] = X_plus2[c] - sbar[c];
        }
        let mut dS = vec![f64::NAN; N + 1];
        for j in 1..N + 1 {
            dS[j] = sbar[j] - sbar[j - 1];
        }
        let X: Vec<f64> = X_all[..N + 1].to_vec();
        let mut X3 = vec![0.0; N + 1];
        for j in 0..N + 1 {
            X3[j] = X[j] * X[j] * X[j];
        }
        dV_all.truncate(N);
        Geometry {
            X,
            X_xi: X_xi_all[..N + 1].to_vec(),
            dV: dV_all,
            dV_xi,
            sbar,
            dS,
            dX,
            Xm,
            X2,
            X3,
            s_in,
            s_out,
        }
    }
}

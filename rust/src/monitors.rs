//! Two of the per-step monitors, as the Python forms them: the cells' emptying rates (`monitors.emptying_rates`) and
//! the near-zone monitors (`horizon.near_zone`), each with the same operations in the same order.

/// Each retained cell's `K_c - 2 + 3 alpha` (`emptying_rates`, with the kernels on): its loss rate through both faces
/// over its content, less the source, NaN below the excision face.
///
/// The loss is summed as numpy sums it, from zero: through the outer face (every cell but the last), then through the
/// inner face (every cell but the origin's), then the last cell's outflow `max(F_N, 0)`.
#[allow(clippy::too_many_arguments)]
pub fn emptying_rates(
    Lam_plus: &[f64],
    Lam_minus: &[f64],
    v_L: &[f64],
    v_R: &[f64],
    rho_L: &[f64],
    rho_R: &[f64],
    X2: &[f64],
    E: &[f64],
    F_N: f64,
    energy_source_rate: f64,
    j_e: usize,
) -> Vec<f64> {
    let N = E.len();
    let A = |j: usize| Lam_plus[j] * (v_L[j] - Lam_minus[j]) / (Lam_plus[j] - Lam_minus[j]);
    let B = |j: usize| -Lam_minus[j] * (Lam_plus[j] - v_R[j]) / (Lam_plus[j] - Lam_minus[j]);
    let outflow = if 0.0 > F_N { 0.0 } else { F_N }; // Python's `max(F_N, 0.0)`, which keeps F_N unless 0 is larger
    let mut rates = vec![f64::NAN; N];
    for c in j_e..N {
        let mut loss = 0.0;
        if c < N - 1 {
            loss += X2[c + 1] * A(c + 1) * rho_L[c + 1]; // through the outer face of cell c, its own value rho^+
        }
        if c >= j_e.max(1) {
            loss += X2[c] * B(c) * rho_R[c]; // through its inner face, rho^-; the origin carries none
        }
        if c == N - 1 {
            loss += outflow;
        }
        rates[c] = loss / E[c] - energy_source_rate;
    }
    rates
}

/// `np.argmin` over `values`: the first NaN if there is one, otherwise the first smallest entry.
pub fn argmin(values: &[f64]) -> usize {
    let mut best = 0;
    for (k, v) in values.iter().enumerate() {
        if v.is_nan() {
            return k;
        }
        if *v < values[best] {
            best = k;
        }
    }
    best
}

/// The interval of the increasing `xp` holding `x` (`horizon.bracket`): the last `j` with `xp[j] <= x`, and whether
/// `x` lies strictly inside `xp[j] .. xp[j + 1]`.
pub fn bracket(x: f64, xp: &[f64]) -> (usize, bool) {
    let j = xp.partition_point(|v| *v <= x) - 1;
    (j, j < xp.len() - 1 && xp[j] != x)
}

/// The straight line through `(xp[j], f0)` and `(xp[j + 1], f1)` at `x`, or `f0` on the grid point
/// (`horizon.linear`): `slope (x - xp[j]) + f0`, unfused.
pub fn linear(x: f64, xp: &[f64], j: usize, inside: bool, f0: f64, f1: f64) -> f64 {
    if !inside {
        return f0;
    }
    let slope = (f1 - f0) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + f0
}

/// `fp` interpolated linearly at `x` (`horizon.interpolate`).
pub fn interpolate(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let (j, inside) = bracket(x, xp);
    linear(x, xp, j, inside, fp[j], if inside { fp[j + 1] } else { f64::NAN })
}

/// The near-zone monitors (`horizon.near_zone`) on the retained cells and faces: at each label radius in `radii`,
/// `e^phi`, `U / Gammabar` and `rho`, NaN off the retained grid; and the cell of the smallest lapse.
#[allow(clippy::too_many_arguments)]
pub fn near_zone(
    Xm: &[f64],
    X: &[f64],
    ephi: &[f64],
    rho: &[f64],
    U: &[f64],
    Gammabar2: &[f64],
    radii: &[f64],
) -> (Vec<f64>, usize) {
    let mut values = vec![f64::NAN; 3 * radii.len()];
    for (n, &R) in radii.iter().enumerate() {
        if Xm[0] <= R && R <= Xm[Xm.len() - 1] {
            values[3 * n] = interpolate(R, Xm, ephi);
            values[3 * n + 2] = interpolate(R, Xm, rho);
        }
        if X[0] <= R && R <= X[X.len() - 1] {
            let (j, inside) = bracket(R, X);
            let v = |i: usize| U[i] / Gammabar2[i].sqrt();
            let v1 = if inside { v(j + 1) } else { v(j) };
            values[3 * n + 1] = linear(R, X, j, inside, v(j), v1);
        }
    }
    (values, argmin(ephi))
}

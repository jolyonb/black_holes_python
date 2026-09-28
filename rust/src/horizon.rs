//! The numbers of the horizon finder (`pbh/horizon.py`, `trapping` and `crossing`; paper Section 8.2,
//! eq:numbh:finder): the trapping function `h_j = U_j + Gammabar_j` on the retained faces, every sign change with its
//! root by the cubic through four faces, and the trapping margins `1 + U / Gammabar` over the grid and over the core.
//!
//! What needs the map, the radius at each root's label, and the masses formed from it, stay in the Python.

/// One sign change of `h` between faces `j` and `j + 1`: the root's fraction `t` of the cell, and whether the trapped
/// side is inside (`h_j < 0`), making it the outer boundary of a trapped region.
pub struct Crossing {
    pub j: usize,
    pub t: f64,
    pub outer: bool,
}

/// The trapping function and what the finder reads from it (`horizon.Trapping`).
pub struct Trapping {
    /// `U + Gammabar` at the faces, NaN below the excision face.
    pub h: Vec<f64>,
    /// How many retained faces are trapped.
    pub trapped_faces: usize,
    /// Every sign change, from the origin outward.
    pub crossings: Vec<Crossing>,
    /// The smallest `1 + U / Gammabar` over the retained faces, and where.
    pub margin: f64,
    pub margin_face: usize,
    /// The same over the core, the central infall region, and where.
    pub core_margin: f64,
    pub core_margin_face: usize,
    /// Whether face `N` is trapped.
    pub outer_face_trapped: bool,
}

/// The trapping function on the retained faces `j_e..N` and what the finder reads from it (`horizon.trapping`), from
/// the face velocities `U` and `Gammabar^2` (full length, `N + 1`).
///
/// `Err` carries numpy's message where `np.nanargmin` raises, on margins that are all NaN.
pub fn trapping(U: &[f64], Gammabar2: &[f64], j_e: usize) -> Result<Trapping, &'static str> {
    let N = U.len() - 1;
    let mut h = vec![f64::NAN; N + 1];
    let mut margin = vec![f64::NAN; N + 1];
    let mut trapped = vec![false; N + 1];
    let mut trapped_faces = 0;
    for j in j_e..N + 1 {
        let Gammabar = Gammabar2[j].sqrt();
        h[j] = U[j] + Gammabar;
        margin[j] = 1.0 + U[j] / Gammabar;
        trapped[j] = h[j] < 0.0;
        trapped_faces += trapped[j] as usize;
    }
    let mut crossings = Vec::new();
    for j in j_e..N {
        if trapped[j] != trapped[j + 1] {
            crossings.push(Crossing {
                j,
                t: crossing(&h, j, j_e, N),
                outer: trapped[j],
            });
        }
    }
    let margin_face = nanargmin(&margin, 0, N + 1)?;
    // The central infall region: from the first retained face out to the first expanding face, at least one face.
    let mut core_end = N + 1 - j_e;
    for j in j_e..N + 1 {
        if U[j] > 0.0 {
            core_end = (j - j_e).max(1);
            break;
        }
    }
    let core_margin_face = nanargmin(&margin, j_e, j_e + core_end)?;
    Ok(Trapping {
        trapped_faces,
        crossings,
        margin: margin[margin_face],
        margin_face,
        core_margin: margin[core_margin_face],
        core_margin_face,
        outer_face_trapped: trapped[N],
        h,
    })
}

/// `np.nanargmin` over `values[start..end]`, as an index into `values`: the first smallest entry that is not NaN.
fn nanargmin(values: &[f64], start: usize, end: usize) -> Result<usize, &'static str> {
    let mut best: Option<usize> = None;
    for k in start..end {
        if values[k].is_nan() {
            continue;
        }
        match best {
            Some(b) if values[k] >= values[b] => {}
            _ => best = Some(k),
        }
    }
    best.ok_or("All-NaN slice encountered")
}

/// Where `h` crosses zero between faces `j` and `j + 1`, as a fraction `t` of the cell (`horizon.crossing`,
/// eq:numbh:finder): the root in `[0, 1]` of the cubic through `h` at faces `j - 1 .. j + 2`, placed at
/// `t = -1 .. 2`, by Newton's method from the linear root, kept inside the bracket `[0, 1]` by bisecting where a step
/// would leave it; the linear root where face `j - 1` or `j + 2` lies outside the retained faces `first .. last`.
pub fn crossing(h: &[f64], j: usize, first: usize, last: usize) -> f64 {
    let linear = -h[j] / (h[j + 1] - h[j]);
    if j < first + 1 || j + 2 > last {
        return linear;
    }
    let (hm, h0, h1, h2) = (h[j - 1], h[j], h[j + 1], h[j + 2]);
    // p(t) = h0 + a1 t + a2 t^2 + a3 t^3, the cubic through (-1, hm), (0, h0), (1, h1), (2, h2)
    let a1 = -hm / 3.0 - h0 / 2.0 + h1 - h2 / 6.0;
    let a2 = hm / 2.0 - h0 + h1 / 2.0;
    let a3 = -hm / 6.0 + h0 / 2.0 - h1 / 2.0 + h2 / 6.0;
    let (mut lo, mut hi) = (0.0, 1.0); // p(lo) has the sign of h0, p(hi) that of h1
    let mut t = linear;
    for _ in 0..60 {
        let p = h0 + t * (a1 + t * (a2 + t * a3));
        if p == 0.0 {
            break;
        }
        if (p < 0.0) == (h0 < 0.0) {
            lo = t;
        } else {
            hi = t;
        }
        let slope = a1 + t * (2.0 * a2 + 3.0 * t * a3);
        let step = if slope != 0.0 { t - p / slope } else { f64::NAN };
        let t_new = if lo < step && step < hi { step } else { 0.5 * (lo + hi) };
        if (t_new - t).abs() <= 1e-15 {
            t = t_new;
            break;
        }
        t = t_new;
    }
    t
}

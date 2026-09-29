"""The horizon finder (paper Section 8.2): every marginally trapped sphere on a slice, and the trapping margin.

The trapping function is `h_j = U_j + Gammabar_j` at the retained faces, both quantities every stage computes: a
face is trapped where `h_j < 0`. The paper's finder takes the outermost sign change from trapped to untrapped and
places the root by the cubic through the four faces around it, in the label,

    j_* = max { j : h_j < 0 <= h_(j+1) },   x_AH = x_(j_*) + t / N,   p(t) = 0,  0 <= t <= 1,
    M_AH / R_H = (1/2) e^(alpha xi) X(xi, x_AH),                                             eq:numbh:finder

with `p` the cubic through `h` at faces `j_* - 1 .. j_* + 2` (`t` = -1..2), and the radius from the analytic map at
that label. The cubic is for the read-out. A linear root errs by an amount that depends on where the root falls
between faces, with a kink wherever the horizon crosses a face, so `M_AH(xi)` carries a sawtooth that the rate fit of
the read-out differentiates. The cubic's interpolation error is fourth order; the root is still only as accurate as
the face values of `h`, which carry the state's second-order error, but that error is smooth in time. Where the four
faces are not all retained, beside the first retained face or the outer face, the root is linear.

Here every sign change is reported, not only the outermost: a collapse can form more than one trapped region, and a new
one appearing outside the excision face is an event (the multi-scale decision), so the finder returns the whole list,
each sphere marked as the outer boundary of a trapped region (an apparent horizon) or the inner one. The apparent
horizon of the paper is the outermost outer boundary. An empty list is the normal state before formation, not an error;
a trapped outer face is an abort.

The finder also returns the margin `1 + U / Gammabar`, which is `-h / Gammabar` shifted so that it crosses zero
exactly where a face becomes trapped, read every step: the flag of formation. It is not a continuous observable of
the threshold, since near threshold the core approaches the critical solution, whose compactness stays well below
one (`collapse.py`). Its minimum over the whole grid can sit in an infalling shell far from the centre, so the
minimum over the core is reported separately: the core is the central infall region, the faces from the first
retained one outward while `U <= 0`, which follows the collapsing core down to any scale and contains a trapped
shell that forms around it. Where the centre expands the core is the first retained face alone. (A core bounded
by the half-central-density radius shrank to the innermost cells in the central runaway and missed the shell.)

On a state whose trapping function is exact at the faces the root is fourth order in the cell width (second order
for the linear root); on an evolved state it is second order, the state's own accuracy.

After formation the horizon table also carries two of the three monitors of Section 8.5 that say whether the
transient is over (the third, the outflow margin at the excision face, is among the face's columns): the near zone,
`(e^phi, U / Gammabar, rho)` at the apparent horizon, `2 M_AH`, and at the sonic radius of the Michel flow, `3 M_AH`
for radiation, whose steady values are `0.620, -1, 6.75` and `0.707, -0.577, 4.00` (Table tab:exc:michel); and the
minimum lapse on the grid, which says the run is still inside the formulation. At the horizon `U / Gammabar = -1`
exactly, since that is where the finder puts it, so there only the lapse and the compression test the flow; the
horizon is chosen over a radius further in because the excision face follows it at `0.7` of its label, `1.4 M`, and
no radius inside that can be read. The radius `R = k M_AH`, in units of `R_H`, is the label radius
`X = k M_AH e^(-alpha xi)`; the cell fields are interpolated linearly between cell midpoints, the ratio between faces,
and a radius off the retained grid gives NaN.

Once excised, the table carries the face's monitors and the fold monitor's evaluation (`FaceValues`, `FoldValues`),
which `excision.py` forms: between the face and `2 M_AH`, the fronts detected, the largest density jump, and for the
worst front (or, with none, the largest jump) its compression, the pre-shock state, its Taub velocity, the threshold of
the boost law and their ratio.
"""

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from pbh.derived import Derived
from pbh.eos import Background, EquationOfState
from pbh.geometry import Geometry
from pbh.layout import Layout
from pbh.maps import Map
from pbh.state import State
from pbh.timestep import Engine
from pbh.types import FloatArray, nan_array


@dataclass(frozen=True)
class Horizon:
    """One marginally trapped sphere: the sign change of `h` between faces `j` and `j + 1`.

    Attributes:
        j: The face inside the crossing.
        x: The interpolated label of the sphere.
        X: Its scaled areal radius, from the analytic map at `x`.
        outer: Whether the trapped region lies inside, `h_j < 0 <= h_(j+1)`, so that this is an apparent horizon;
            otherwise the trapped region lies outside and this is its inner boundary.
    """

    j: int
    x: float
    X: float
    outer: bool


@dataclass(frozen=True)
class HorizonReport:
    """What the finder reports about one slice.

    Attributes:
        h: The trapping function `U + Gammabar` at the faces, NaN below the excision face.
        trapped_faces: How many retained faces are trapped.
        horizons: Every marginally trapped sphere, from the origin outward.
        apparent: The apparent horizon, the outermost outer boundary; `None` if no face is trapped.
        M_AH: The apparent-horizon mass in units of `R_H`, eq:numbh:finder; NaN if none.
        residual: `2m/R - 1` at the apparent horizon, with the cumulative mass interpolated there; NaN if none.
        margin: The smallest `1 + U / Gammabar` over the retained faces, and `margin_face` where.
        core_margin: The same over the core, the central infall region (`U <= 0` from the first retained face), and
            where.
        outer_face_trapped: Whether face `N` is trapped, which aborts the run.
    """

    h: FloatArray
    trapped_faces: int
    horizons: tuple[Horizon, ...]
    apparent: Horizon | None
    M_AH: float
    residual: float
    margin: float
    margin_face: int
    core_margin: float
    core_margin_face: int
    outer_face_trapped: bool


def crossing(h: FloatArray, j: int, first: int, last: int) -> float:
    """Where `h` crosses zero between faces `j` and `j + 1`, as a fraction `t` of the cell (eq:numbh:finder).

    The root in `[0, 1]` of the cubic through `h` at faces `j - 1 .. j + 2`, placed at `t = -1 .. 2`, found by
    Newton's method from the linear root and kept inside the bracket `[0, 1]` that the sign change guarantees, a step
    that would leave it bisecting instead. The linear root where face `j - 1` or `j + 2` lies outside the retained faces
    `first .. last`.
    """
    linear = -h[j] / (h[j + 1] - h[j])
    if j - 1 < first or j + 2 > last:
        return float(linear)
    hm, h0, h1, h2 = (float(v) for v in h[j - 1 : j + 3])
    # p(t) = h0 + a1 t + a2 t^2 + a3 t^3, the cubic through (-1, hm), (0, h0), (1, h1), (2, h2)
    a1 = -hm / 3.0 - h0 / 2.0 + h1 - h2 / 6.0
    a2 = hm / 2.0 - h0 + h1 / 2.0
    a3 = -hm / 6.0 + h0 / 2.0 - h1 / 2.0 + h2 / 6.0
    lo, hi = 0.0, 1.0  # p(lo) has the sign of h0, p(hi) that of h1
    t = float(linear)
    for _ in range(60):
        p = h0 + t * (a1 + t * (a2 + t * a3))
        if p == 0.0:
            break
        if (p < 0.0) == (h0 < 0.0):
            lo = t
        else:
            hi = t
        slope = a1 + t * (2.0 * a2 + 3.0 * t * a3)
        step = t - p / slope if slope != 0.0 else math.nan
        t_new = step if lo < step < hi else 0.5 * (lo + hi)
        if abs(t_new - t) <= 1e-15:
            t = t_new
            break
        t = t_new
    return t


@dataclass(frozen=True)
class Trapping:
    """The finder's numbers on one slice, before the map places its roots (`trapping`).

    Attributes:
        h: The trapping function `U + Gammabar` at the faces, NaN below the excision face.
        trapped_faces: How many retained faces are trapped.
        crossings: Every sign change `(j, t, outer)` from the origin outward: between faces `j` and `j + 1`, at the
            fraction `t` of the cell (`crossing`), `outer` if the trapped side is inside.
        margin, margin_face, core_margin, core_margin_face, outer_face_trapped: As in `HorizonReport`.
    """

    h: FloatArray
    trapped_faces: int
    crossings: tuple[tuple[int, float, bool], ...]
    margin: float
    margin_face: int
    core_margin: float
    core_margin_face: int
    outer_face_trapped: bool


def trapping(U: FloatArray, Gammabar2: FloatArray, layout: Layout) -> Trapping:
    """The trapping function on the retained faces, its sign changes with their roots, and the margins.

    The Rust engine computes the same numbers with the same operations (`pbh.rust_engine.trapping`).
    """
    N, j_e = layout.N, layout.j_e
    faces = layout.faces
    Gammabar = np.sqrt(Gammabar2[faces])
    h = nan_array(N + 1)
    h[faces] = U[faces] + Gammabar
    trapped = h[faces] < 0.0
    crossings = tuple(
        (j_e + int(k), crossing(h, j_e + int(k), j_e, N), bool(trapped[k]))
        for k in np.flatnonzero(trapped[:-1] != trapped[1:])
    )
    margin = nan_array(N + 1)
    margin[faces] = 1.0 + U[faces] / Gammabar
    k = int(np.nanargmin(margin))
    expanding = np.flatnonzero(U[faces] > 0.0)
    core_end = max(int(expanding[0]), 1) if expanding.size else N + 1 - j_e  # the central infall region, at least one
    k_core = int(np.nanargmin(margin[j_e : j_e + core_end])) + j_e
    return Trapping(
        h=h,
        trapped_faces=int(np.sum(trapped)),
        crossings=crossings,
        margin=float(margin[k]),
        margin_face=k,
        core_margin=float(margin[k_core]),
        core_margin_face=k_core,
        outer_face_trapped=bool(trapped[-1]),
    )


def find_horizons(
    state: State,
    d: Derived,
    geo: Geometry,
    bg: Background,
    eos: EquationOfState,
    map: Map,
    layout: Layout,
    xi: float,
    engine: Engine = Engine.PYTHON,
) -> HorizonReport:
    """Evaluate the trapping function on the retained faces and report every sign change and the margins.

    `engine` computes `trapping`, the finder's numbers; the radius at each root comes from the map, here.
    """
    N = layout.N
    if engine is Engine.RUST:
        from pbh import rust_engine  # loaded only on the Rust engine, as in `Scheme`

        t = rust_engine.trapping(state.U, d.Gammabar2, layout)
    else:
        t = trapping(state.U, d.Gammabar2, layout)

    horizons: list[Horizon] = []
    for j, fraction, outer in t.crossings:
        x = (j + fraction) / N
        X = map.radius(xi, x)
        horizons.append(Horizon(j=j, x=x, X=X, outer=outer))
    outer_boundaries = [horizon for horizon in horizons if horizon.outer]
    apparent = outer_boundaries[-1] if outer_boundaries else None

    M_AH, residual = float("nan"), float("nan")
    if apparent is not None:
        M_AH = 0.5 * float(np.exp(eos.alpha_float * xi)) * apparent.X
        j = apparent.j
        M_at = d.M[j] + (apparent.x * N - j) * (d.M[j + 1] - d.M[j])  # the cumulative mass at the horizon's label
        residual = M_at / (apparent.X * bg.Gammabar2) - 1.0  # 2m/R = M / (X Gammabar_FRW^2) in the scaled variables

    return HorizonReport(
        h=t.h,
        trapped_faces=t.trapped_faces,
        horizons=tuple(horizons),
        apparent=apparent,
        M_AH=M_AH,
        residual=residual,
        margin=t.margin,
        margin_face=t.margin_face,
        core_margin=t.core_margin,
        core_margin_face=t.core_margin_face,
        outer_face_trapped=t.outer_face_trapped,
    )


@dataclass(frozen=True)
class HorizonRow:
    """The row of the horizon table, one per step (output specification, Section 8)."""

    step: int
    xi: float
    trapped_faces: int
    horizons: int
    """How many marginally trapped spheres the slice has."""
    j_star: int
    """The face inside the apparent horizon; `-1` if none."""
    x_AH: float
    X_AH: float
    M_AH: float
    residual: float
    margin: float
    margin_face: int
    core_margin: float
    core_margin_face: int
    zone_ratio: float
    """`x_AH / (x_t - Delta_t)` of the outermost pinned zone: how close the horizon is to the transition; NaN without
    a zone or a horizon."""
    # the excision face (Section 8.3), NaN or -1 while unexcised
    j_e: int
    mu: float
    sound_margin: float
    a_over_Theta: float
    Lambda_plus: float
    h_e: float
    h_e1: float
    h_e2: float
    faces_to_horizon: int
    M_e: float
    F_e: float
    R_e_over_M_AH: float
    physical_margin: float
    # the fold monitor (Section 8.3, `FoldValues`), NaN, 0 or -1 while unexcised
    fold_X_zone: float
    fold_fronts: int
    fold_jump: float
    fold_cell: int
    fold_X: float
    fold_compression: float
    fold_U1: float
    fold_Gammabar1: float
    fold_v12: float
    fold_threshold: float
    fold_ratio: float
    # the near zone and the lapse (Section 8.5)
    lapse_AH: float
    v_AH: float
    rho_AH: float
    lapse_sonic: float
    v_sonic: float
    rho_sonic: float
    min_lapse: float
    min_lapse_X: float

    @classmethod
    def of(
        cls,
        step: int,
        xi: float,
        report: HorizonReport,
        zone_inner_edge: float | None,
        near: NearZone,
        face: FaceValues | None = None,
    ) -> HorizonRow:
        """The row for a step's report, with the near-zone monitors and the face's monitors once excised."""
        a = report.apparent
        ratio = a.x / zone_inner_edge if a is not None and zone_inner_edge is not None else float("nan")
        f = face if face is not None else UNEXCISED
        fold = f.fold
        return cls(
            step=step,
            xi=xi,
            trapped_faces=report.trapped_faces,
            horizons=len(report.horizons),
            j_star=a.j if a is not None else -1,
            x_AH=a.x if a is not None else float("nan"),
            X_AH=a.X if a is not None else float("nan"),
            M_AH=report.M_AH,
            residual=report.residual,
            margin=report.margin,
            margin_face=report.margin_face,
            core_margin=report.core_margin,
            core_margin_face=report.core_margin_face,
            zone_ratio=ratio,
            j_e=f.j_e,
            mu=f.mu,
            sound_margin=f.sound_margin,
            a_over_Theta=f.a_over_Theta,
            Lambda_plus=f.Lambda_plus,
            h_e=f.h[0],
            h_e1=f.h[1],
            h_e2=f.h[2],
            faces_to_horizon=f.faces_to_horizon,
            M_e=f.M_e,
            F_e=f.F_e,
            R_e_over_M_AH=f.R_e_over_M_AH,
            physical_margin=f.physical_margin,
            fold_X_zone=fold.X_zone,
            fold_fronts=fold.fronts,
            fold_jump=fold.jump,
            fold_cell=fold.cell,
            fold_X=fold.X,
            fold_compression=fold.compression,
            fold_U1=fold.U1,
            fold_Gammabar1=fold.Gammabar1,
            fold_v12=fold.v12,
            fold_threshold=fold.threshold,
            fold_ratio=fold.ratio,
            **{f.name: getattr(near, f.name) for f in dataclasses.fields(near)},  # floats: no deep copy
        )


HORIZON = 2.0
"""The inner radius of the near-zone monitor, the apparent horizon, in units of the apparent-horizon mass."""


@dataclass(frozen=True)
class NearZone:
    """The near-zone monitors of one slice (module docstring), NaN without an apparent horizon.

    Attributes:
        lapse_AH, v_AH, rho_AH: `e^phi`, `U / Gammabar` and `rho` at the apparent horizon, `2 M_AH`.
        lapse_sonic, v_sonic, rho_sonic: The same at the sonic radius of the Michel flow.
        min_lapse: The smallest cell lapse on the grid, at every step, and `min_lapse_X` the midpoint of its cell.
    """

    lapse_AH: float
    v_AH: float
    rho_AH: float
    lapse_sonic: float
    v_sonic: float
    rho_sonic: float
    min_lapse: float
    min_lapse_X: float


def near_zone(
    state: State,
    d: Derived,
    geo: Geometry,
    report: HorizonReport,
    eos: EquationOfState,
    layout: Layout,
    xi: float,
    engine: Engine = Engine.PYTHON,
) -> NearZone:
    """The near-zone monitors on this slice; the minimum lapse whether or not a horizon has formed.

    `engine` forms the numbers (`near_zone_numbers`, or the Rust engine's with the same operations).
    """
    cells, faces = layout.cells, layout.faces
    radii: list[float] = []
    if report.apparent is not None:
        scale = report.M_AH * float(np.exp(-eos.alpha_float * xi))  # the label radius of R = M_AH
        radii = [radius * scale for radius in (HORIZON, eos.sonic_radius_over_mass)]
    fields = (geo.Xm[cells], geo.X[faces], d.ephi[cells], d.rho[cells], state.U[faces], d.Gammabar2[faces], radii)
    if engine is Engine.RUST:
        from pbh import rust_engine  # loaded only on the Rust engine, as in `Scheme`

        values, k = rust_engine.near_zone(*fields)
    else:
        values, k = near_zone_numbers(*fields)
    values = values if radii else [float("nan")] * 6
    k += layout.j_e
    return NearZone(*values, min_lapse=float(d.ephi[k]), min_lapse_X=float(geo.Xm[k]))


def near_zone_numbers(
    Xm: FloatArray,
    X: FloatArray,
    ephi: FloatArray,
    rho: FloatArray,
    U: FloatArray,
    Gammabar2: FloatArray,
    radii: list[float],
) -> tuple[list[float], int]:
    """The near-zone monitors' numbers from the retained cells and faces.

    `(e^phi, U / Gammabar, rho)` at each label radius, NaN off the retained grid, and the retained cell of the
    smallest lapse; the Rust engine forms the same (`rust_engine.near_zone`).
    """
    values = [float("nan")] * (3 * len(radii))
    for n, R in enumerate(radii):
        if Xm[0] <= R <= Xm[-1]:
            values[3 * n] = interpolate(R, Xm, ephi)
            values[3 * n + 2] = interpolate(R, Xm, rho)
        if X[0] <= R <= X[-1]:
            j, inside = bracket(R, X)  # U / Gammabar at the faces around R: at one face if R is on it
            v = [float(U[i]) / math.sqrt(float(Gammabar2[i])) for i in ((j, j + 1) if inside else (j,))]
            values[3 * n + 1] = linear(R, X, j, inside, (v[0], v[-1]))
    return values, int(np.argmin(ephi))


def bracket(x: float, xp: FloatArray) -> tuple[int, bool]:
    """The interval of the increasing `xp` holding `x`, `xp[0] <= x <= xp[-1]`.

    The last `j` with `xp[j] <= x`, and whether `x` lies strictly inside `xp[j] .. xp[j + 1]` (otherwise it is on the
    grid point `xp[j]`).
    """
    j = int(np.searchsorted(xp, x, side="right")) - 1
    return j, j < len(xp) - 1 and float(xp[j]) != x


def linear(x: float, xp: FloatArray, j: int, inside: bool, f: tuple[float, float]) -> float:
    """The straight line through `(xp[j], f[0])` and `(xp[j + 1], f[1])` at `x`, or `f[0]` on the grid point.

    `slope (x - xp[j]) + f[0]`, the formula of `np.interp`, written out so that both engines form it alike: whether
    `np.interp`'s compiled multiply-add is fused depends on the platform (it is, on macOS arm64), and the Rust engine
    does not fuse.
    """
    if not inside:
        return f[0]
    x_j = float(xp[j])
    slope = (f[1] - f[0]) / (float(xp[j + 1]) - x_j)
    return slope * (x - x_j) + f[0]


def interpolate(x: float, xp: FloatArray, fp: FloatArray) -> float:
    """`fp` interpolated linearly at `x`, `xp[0] <= x <= xp[-1]` (`linear`, on the bracketing interval)."""
    j, inside = bracket(x, xp)
    return linear(x, xp, j, inside, (float(fp[j]), float(fp[j + 1]) if inside else math.nan))


NAN = float("nan")


@dataclass(frozen=True)
class FoldValues:
    """The fold monitor of one excised slice (Section 8.3, `excision.fold_monitor`), the `fold_` columns of the row.

    The front reported is the detected front of largest `ratio`, the one an abort names; with none detected, the
    cell of the largest jump, measured as if it were one, so that every row says how near the slice came.

    Attributes:
        X_zone: The label radius of `R = 2 M_AH`, the zone's outer end.
        fronts: The fronts detected in the zone: runs of adjacent cells whose jump exceeds the detection level.
        jump: The largest jump `rho_(c-1) / rho_(c+1)` across a cell `c` of the zone, the denser side inside; NaN if
            the zone has no such cell.
        cell: The cell of the reported front's largest jump, `-1` if the zone has none; `X` its midpoint label radius.
        compression: The front's compression `rho_2 / rho_1`, measured across it.
        U1: `Utilde_1`, the velocity at the pre-shock face, and `Gammabar1` there.
        v12: The relative velocity across the front, from the compression by the Taub relation eq:eul:taubeta.
        threshold: `Gammabar_1 / |Utilde_1|` of eq:eul:boost, infinite where `Utilde_1 >= 0`.
        ratio: `v12 / threshold`; a detected front at one or above folds the slice, and the run ends.
    """

    X_zone: float
    fronts: int
    jump: float
    cell: int
    X: float
    compression: float
    U1: float
    Gammabar1: float
    v12: float
    threshold: float
    ratio: float


NO_FOLD = FoldValues(NAN, 0, NAN, -1, NAN, NAN, NAN, NAN, NAN, NAN, NAN)
"""The fold columns while unexcised."""


@dataclass(frozen=True)
class FaceValues:
    """The excision face's monitors as the horizon table stores them, the fold monitor's among them.

    `excision.py` produces them.
    """

    j_e: int
    mu: float
    sound_margin: float
    a_over_Theta: float
    Lambda_plus: float
    h: tuple[float, float, float]
    faces_to_horizon: int
    M_e: float
    F_e: float
    R_e_over_M_AH: float
    physical_margin: float
    fold: FoldValues = NO_FOLD


UNEXCISED = FaceValues(-1, NAN, NAN, NAN, NAN, (NAN, NAN, NAN), -1, NAN, NAN, NAN, NAN)
"""The face columns while there is no excision face."""

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
from pbh.types import FloatArray


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


def find_horizons(
    state: State, d: Derived, geo: Geometry, bg: Background, eos: EquationOfState, map: Map, layout: Layout, xi: float
) -> HorizonReport:
    """Evaluate the trapping function on the retained faces and report every sign change and the margins."""
    N, j_e = layout.N, layout.j_e
    faces = layout.faces
    Gammabar = np.sqrt(d.Gammabar2[faces])
    h = np.full(N + 1, np.nan)
    h[faces] = state.U[faces] + Gammabar
    trapped = h[faces] < 0.0
    retained = np.arange(j_e, N + 1)

    horizons: list[Horizon] = []
    for k in np.flatnonzero(trapped[:-1] != trapped[1:]):
        j = int(retained[k])
        x = (j + crossing(h, j, j_e, N)) / N
        X = float(map.radius_at(xi, np.array([x]))[0])
        horizons.append(Horizon(j=j, x=x, X=X, outer=bool(trapped[k])))
    outer_boundaries = [horizon for horizon in horizons if horizon.outer]
    apparent = outer_boundaries[-1] if outer_boundaries else None

    M_AH, residual = float("nan"), float("nan")
    if apparent is not None:
        M_AH = 0.5 * float(np.exp(float(eos.alpha) * xi)) * apparent.X
        j = apparent.j
        M_at = d.M[j] + (apparent.x * N - j) * (d.M[j + 1] - d.M[j])  # the cumulative mass at the horizon's label
        residual = M_at / (apparent.X * bg.Gammabar2) - 1.0  # 2m/R = M / (X Gammabar_FRW^2) in the scaled variables

    margin = np.full(N + 1, np.nan)
    margin[faces] = 1.0 + state.U[faces] / Gammabar
    k = int(np.nanargmin(margin))
    expanding = np.flatnonzero(state.U[faces] > 0.0)
    core_end = max(int(expanding[0]), 1) if expanding.size else N + 1 - j_e  # the central infall region, at least one
    k_core = int(np.nanargmin(margin[j_e : j_e + core_end])) + j_e

    return HorizonReport(
        h=h,
        trapped_faces=int(np.sum(trapped)),
        horizons=tuple(horizons),
        apparent=apparent,
        M_AH=M_AH,
        residual=residual,
        margin=float(margin[k]),
        margin_face=k,
        core_margin=float(margin[k_core]),
        core_margin_face=k_core,
        outer_face_trapped=bool(trapped[-1]),
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
            **dataclasses.asdict(near),
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
    state: State, d: Derived, geo: Geometry, report: HorizonReport, eos: EquationOfState, layout: Layout, xi: float
) -> NearZone:
    """The near-zone monitors on this slice; the minimum lapse whether or not a horizon has formed."""
    cells, faces = layout.cells, layout.faces
    k = int(np.argmin(d.ephi[cells])) + layout.j_e
    values = [float("nan")] * 6
    if report.apparent is not None:
        Xm, X = geo.Xm[cells], geo.X[faces]
        v = state.U[faces] / np.sqrt(d.Gammabar2[faces])
        scale = report.M_AH * float(np.exp(-float(eos.alpha) * xi))  # the label radius of R = M_AH
        for n, radius in enumerate((HORIZON, eos.sonic_radius_over_mass)):
            R = radius * scale
            if Xm[0] <= R <= Xm[-1]:
                values[3 * n] = float(np.interp(R, Xm, d.ephi[cells]))
                values[3 * n + 2] = float(np.interp(R, Xm, d.rho[cells]))
            if X[0] <= R <= X[-1]:
                values[3 * n + 1] = float(np.interp(R, X, v))
    return NearZone(*values, min_lapse=float(d.ephi[k]), min_lapse_X=float(geo.Xm[k]))


@dataclass(frozen=True)
class FaceValues:
    """The excision face's monitors as the horizon table stores them; `excision.py` produces them."""

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


NAN = float("nan")
UNEXCISED = FaceValues(-1, NAN, NAN, NAN, NAN, (NAN, NAN, NAN), -1, NAN, NAN, NAN, NAN)
"""The face columns while there is no excision face."""

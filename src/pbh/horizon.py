"""The horizon finder (paper Section 8.2): every marginally trapped sphere on a slice, and the trapping margin.

The trapping function is `h_j = U_j + Gammabar_j` at the retained faces, both quantities every stage computes: a
face is trapped where `h_j < 0`. The paper's finder takes the outermost sign change from trapped to untrapped and
interpolates linearly in the label,

    j_* = max { j : h_j < 0 <= h_(j+1) },   x_AH = x_(j_*) + (1/N) (-h_(j_*)) / (h_(j_*+1) - h_(j_*)),
    M_AH / R_H = (1/2) e^(alpha xi) X(xi, x_AH),                                             eq:numbh:finder

the radius from the analytic map at that label. Here every sign change is reported, not only the outermost: a
collapse can form more than one trapped region, and a new one appearing outside the excision face is an event
(the multi-scale decision), so the finder returns the whole list, each sphere marked as the outer boundary of a
trapped region (an apparent horizon) or the inner one. The apparent horizon of the paper is the outermost outer
boundary. An empty list is the normal state before formation, not an error; a trapped outer face is an abort.

The finder also returns the margin `1 + U / Gammabar`, which is `-h / Gammabar` shifted so that it crosses zero
exactly where a face becomes trapped: the continuous observable of the criticality study, decreasing in the
amplitude of the perturbation and read every step. Its minimum over the whole grid can sit in an outgoing wave far
from the centre, so the minimum over the core, the faces inside the radius where the density has fallen to half
its central value, is reported separately.

On a state whose apparent horizon is known exactly the interpolated radius is second order, below `0.09 Delta X^2`
in the paper's measurement, though the constant depends on where the root falls between faces.
"""

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
        core_margin: The same over the core, the faces inside the half-central-density radius, and where.
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
        fraction = -h[j] / (h[j + 1] - h[j])  # where h crosses zero between the faces, as a fraction of the cell
        x = (j + fraction) / N
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
    half = np.flatnonzero(d.rho[layout.cells] <= 0.5 * d.rho[j_e])
    core_faces = slice(j_e, j_e + int(half[0]) + 1) if half.size else faces  # the faces of the core's cells
    k_core = int(np.nanargmin(margin[core_faces])) + j_e

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

    @classmethod
    def of(
        cls, step: int, xi: float, report: HorizonReport, zone_inner_edge: float | None, face: FaceValues | None = None
    ) -> HorizonRow:
        """The row for a step's report, with the face's monitors once excised."""
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
        )


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

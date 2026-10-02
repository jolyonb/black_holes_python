"""Excision (paper Sections 8.2 and 8.3): the switch-on, dropping the interior, the assertions, re-excision.

Once a horizon has formed, the trapped interior is dropped: the cells inside an excision face `j_e` are never
read again, the face becomes an ordinary evolved face whose four looking-inward rows are the one-sided closure of
`stencils.py`, and the one unknown excision adds, the mass inside the face, obeys eq:numbh:mass with the same flux
as the first retained cell's energy, so that the cumulative-sum bookkeeping stays exact. Nothing is imposed at
the face: every characteristic leaves through it, which is what the switch-on tests and the assertions guarantee.

The switch is thrown at the first step boundary at which a face is trapped and the excision face is admissible.
Its label is the rest-radius rule corrected by the ramp factor (eq:numbh:xe),

    x_e = eta e^(-alpha tau_on) x_AH,   j_e = ceil(N x_e),

and four things are required (Section 8.2): the face lies inside the horizon and outside the origin,
`1 <= j_e < j_*`; the outflow margin `mu` of eq:exc:margin is positive there, the light-cone test, stricter than
the sound condition the stencils need; the faces `j_e`, `j_e + 1` and `j_e + 2` are all trapped, so that every
point the closure's rows touch lies where nothing returns; and the map's transition, placed to span the areal radii
`(c_t -+ c_Delta) X_AH`, ends below the label `0.8`, so that the outer face sits on the static part of the map. The
transition is placed in areal radius, not in label: on a stretched map the labels near the origin are inflated, and a
rule in labels would refuse holes that fit (Section 8.1). If any fails the step
is taken unexcised and the test repeated: the trapped region is a thin shell when it first appears and thickens
inward. A refused attempt is an event, and so is the switch.

Every later step asserts `mu > 0` at the face and the three faces trapped; neither has a cure, since no admissible
move of the face is inward, so a failure ends the run. Outward the face may move at any time: if
`ceil(N eta_r x_AH)` exceeds `j_e`, the face advances to it, or as far as leaves the three faces trapped, and the
mass inside is reset to the cumulative sum at the new face, non-decreasing by construction. The same move takes
the face to a horizon that appears further out, whatever lies between (the multi-scale decision): causality
depends only on the face.

Every excised step also runs the fold monitor (Section 8.3): a front captured between the face and `2 M_AH`, a density
jump above `1.5` across a cell, is held against the threshold `Gammabar_1 / |Utilde_1|` of the boost law eq:eul:boost
ahead of it, with the relative velocity from the front's compression by the Taub relation eq:eul:taubeta. Above it the
slice folds behind the front, the areal radius decreasing outward: the run has left the formulation, and it ends, the
cure being to excise further out. Every completed evaluation is in the horizon table, the aborting one included,
which the driver writes before it ends the run.

A switch-on is repeatable. A second one is triggered when the horizon approaches its zone's transition through
accretion, or jumps past it to a new trapped region, and pins a further zone outside the existing ones; the
four tests apply again, and the new zone must not overlap the old.
"""

import math
from dataclasses import dataclass

import numpy as np

from pbh.config import ExcisionConfig
from pbh.derived import Derived
from pbh.eos import Background, EquationOfState
from pbh.equations import Speeds
from pbh.geometry import Geometry
from pbh.horizon import FaceValues, FoldValues, HorizonReport
from pbh.layout import Layout
from pbh.maps import Map, Zone
from pbh.state import State
from pbh.types import BoolArray, FloatArray

OUTER_STATIC_LABEL = 0.8
"""The transition must end below this label, so that the outer face sits on the static part of the map."""
FOLD_DETECTION = 1.5
"""The fold monitor's detection level: a density jump `rho_(c-1) / rho_(c+1)` across a cell above it is a front."""
FOLD_SPAN = 3
"""The cells to either side of a front over which its compression is measured: with mc on the density and minmod on
the velocity a captured front is monotone in two to three cells (Section 7.7), and three reach its plateaux."""
FOLD_ZONE = 2.0
"""The fold monitor's outer radius, `R = 2 M_AH`, in units of the apparent-horizon mass (Section 8.3)."""
FOLD_CHECK = "the fold monitor v_12 < Gammabar_1/|Utilde_1|"
"""The fold monitor's name in an abort."""


class ExcisionError(Exception):
    """An assertion of the excised scheme failed: the run has left the regime the closure is certified in.

    `detail`, if given, follows the failure in the message: what it means and the cure.
    """

    def __init__(self, check: str, face: int, value: float, detail: str = "") -> None:
        super().__init__(f"{check} failed at face {face}: {value!r}" + (f"; {detail}" if detail else ""))
        self.check = check
        self.face = face
        self.value = value


# --- the switch-on ---


@dataclass(frozen=True)
class SwitchAttempt:
    """One attempt to throw the switch: the candidate face, the four tests, and their values.

    Attributes:
        x_AH: The apparent horizon's label the attempt is based on, and `j_star` its face.
        x_e: The excision label of eq:numbh:xe, and `j_e` the candidate face.
        inside_horizon: Test one, `1 <= j_e < j_star`; for a run already excised the candidate is the current face
            if that lies further out, since the face never moves inward.
        margin_positive: Test two, `mu > 0` at `j_e`, with `mu` the value.
        three_trapped: Test three, the faces `j_e` to `j_e + 2` trapped, with `h` their trapping values.
        transition_fits: Test four, `x_t + Delta_t < 0.8`, with the two labels of the transition, placed to span the
            areal radii `(c_t -+ c_Delta) X_AH`, an extension's pushed outward beyond the last zone; `X_out` is the
            radius where it ends and `X_static` the radius of label `0.8`.
            A failure is final: the horizon only grows.
        no_overlap: With existing zones, the new transition starts beyond the last one's end; an extension is placed
            there if the horizon alone would put it closer in.
    """

    x_AH: float
    j_star: int
    x_e: float
    j_e: int
    inside_horizon: bool
    margin_positive: bool
    mu: float
    three_trapped: bool
    h: tuple[float, float, float]
    transition_fits: bool
    x_t: float
    Delta_t: float
    X_out: float
    X_static: float
    no_overlap: bool

    @property
    def passed(self) -> bool:
        """Whether every test passed."""
        return (
            self.inside_horizon
            and self.margin_positive
            and self.three_trapped
            and self.transition_fits
            and self.no_overlap
        )

    @property
    def failed(self) -> list[str]:
        """The names of the tests that failed."""
        tests = {
            "inside_horizon": self.inside_horizon,
            "margin_positive": self.margin_positive,
            "three_trapped": self.three_trapped,
            "transition_fits": self.transition_fits,
            "no_overlap": self.no_overlap,
        }
        return [name for name, ok in tests.items() if not ok]

    def zone(self, xi_on: float, tau_on: float) -> Zone:
        """The pinned zone this switch-on adds."""
        return Zone(xi_on=xi_on, tau_on=tau_on, x_t=self.x_t, Delta_t=self.Delta_t)


def outflow_margin(j: int, state: State, d: Derived, geo: Geometry, eos: EquationOfState, layout: Layout) -> float:
    """The outflow margin `mu = (alpha X + d_xi X) - alpha <ephi> (U + Gammabar)` of eq:exc:margin at face `j`.

    Evaluated with the closure's own face value of the lapse: at the excision face the first-order rows take the
    cell behind it, `<ephi>_(j_e) = ephi_(j_e)`; at a candidate face on the unexcised grid the same one-sided value
    is used, since that is what the rows will see once the switch is thrown. `mu > 0` is the light-cone criterion
    of the outflow-margin theorem: every signal leaves through the face and none is tangent to it.
    """
    alpha = eos.alpha_float
    ephi_face = d.ephi[j]  # the cell outside face j, which the first-order rows take as the face value
    Gammabar = math.sqrt(d.Gammabar2[j])
    return (alpha * geo.X[j] + geo.X_xi[j]) - alpha * ephi_face * (state.U[j] + Gammabar)


def attempt_switch_on(
    report: HorizonReport,
    state: State,
    d: Derived,
    geo: Geometry,
    eos: EquationOfState,
    layout: Layout,
    excision: ExcisionConfig,
    zones: tuple[Zone, ...],
    grid: Map,
    xi: float,
) -> SwitchAttempt:
    """Run the four tests of Section 8.2 on the apparent horizon of `report`; the caller must have one.

    `grid` is the map the run is on at `xi`, which places the transition's areal radii at their labels.
    """
    apparent = report.apparent
    assert apparent is not None, "a switch-on needs an apparent horizon"
    N = layout.N
    alpha = eos.alpha_float
    x_e = excision.eta * math.exp(-alpha * excision.tau_on) * apparent.x
    j_e = max(math.ceil(N * x_e), layout.j_e)  # a face already further out stays where it is
    inside = 1 <= j_e < apparent.j
    mu = outflow_margin(j_e, state, d, geo, eos, layout) if inside else float("nan")
    h = tuple(float(report.h[j]) if j <= N else float("nan") for j in (j_e, j_e + 1, j_e + 2))
    three = inside and j_e + 2 <= N and all(value < 0.0 for value in h)
    # the transition, spanning the areal radii (c_t -+ c_Delta) X_AH; an extension starts it beyond the last zone's end
    x_in, x_out = (label_of(grid, xi, (excision.c_t + s * excision.c_Delta) * apparent.X) for s in (-1.0, 1.0))
    Delta_t = 0.5 * (x_out - x_in)
    x_t = place_transition(0.5 * (x_in + x_out), Delta_t, zones)
    fits = x_t + Delta_t < OUTER_STATIC_LABEL
    X_out = grid.radius(xi, x_t + Delta_t)  # where it ends, an extension's pushed outward
    no_overlap = not zones or x_t - Delta_t >= zones[-1].outer_edge  # the test `BlendMap` makes, exactly
    return SwitchAttempt(
        x_AH=apparent.x,
        j_star=apparent.j,
        x_e=x_e,
        j_e=j_e,
        inside_horizon=inside,
        margin_positive=bool(inside and mu > 0.0),
        mu=mu,
        three_trapped=three,
        h=(h[0], h[1], h[2]),
        transition_fits=fits,
        x_t=x_t,
        Delta_t=Delta_t,
        X_out=X_out,
        X_static=grid.radius(xi, OUTER_STATIC_LABEL),
        no_overlap=no_overlap,
    )


def label_of(grid: Map, xi: float, X: float) -> float:
    """The label `u` at which the map puts the areal radius `X` at time `xi`, by bisection: every map is increasing.

    Labels beyond the outer face are returned up to `2`, which is enough to tell that a transition does not fit.
    """
    lo, hi = 0.0, 2.0
    if grid.radius(xi, hi) <= X:
        return hi
    while True:
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):  # converged to the last bit
            return hi
        if grid.radius(xi, mid) < X:
            lo = mid
        else:
            hi = mid


def place_transition(x_t: float, Delta_t: float, zones: tuple[Zone, ...]) -> float:
    """The centre of a new zone's transition: `x_t`, or with zones already present no nearer than where the last ends.

    An extension's transition starts at the last zone's outer edge at the least, `x_t - Delta_t >= x_t,k + Delta_t,k`,
    and must satisfy that exactly, as `BlendMap` checks it: `(a + b) - b` can round below `a`, so the centre is
    stepped up by an ulp at a time until it does.
    """
    if not zones:
        return x_t
    edge = zones[-1].outer_edge
    x_t = max(x_t, edge + Delta_t)
    while x_t - Delta_t < edge:
        x_t = math.nextafter(x_t, math.inf)
    return x_t


# --- dropping the interior ---


def excise(state: State, layout: Layout, j_e: int) -> tuple[State, Layout]:
    """Drop the cells inside face `j_e` and set the mass inside it to the cumulative sum there (Section 8.2).

    `M_(j_e) = M_e + 3 sum_(k < j_e) E_k` over the cells being dropped, so the cumulative mass at every retained
    face is unchanged, and the bookkeeping continues exactly. The face may only move outward.
    """
    if j_e <= layout.j_e or j_e >= layout.N:
        raise ValueError(f"the excision face may only move outward within the grid: {layout.j_e} -> {j_e}")
    M_e = state.M_e + 3.0 * float(np.sum(state.E[layout.j_e : j_e]))
    E, U = state.E.copy(), state.U.copy()
    E[:j_e] = np.nan
    U[:j_e] = np.nan
    return State(E=E, U=U, W=state.W, M_e=M_e), Layout(layout.N, j_e=j_e)


# --- every excised step ---


def check_face(
    report: HorizonReport,
    state: State,
    d: Derived,
    speeds: Speeds,
    F_e: float,
    geo: Geometry,
    bg: Background,
    eos: EquationOfState,
    layout: Layout,
    xi: float,
) -> FaceValues:
    """Assert `mu > 0` at the face and the three faces trapped, and collect the face monitors and the fold monitor.

    The values (output specification, Section 8): the light-cone margin `mu`; the outflow margin for sound
    `-(Theta + a)`; `a / |Theta|`, below `sqrt(w)` in the certified regime; `Lambda^+ = max(Theta + a, 0)`, which
    must vanish for the flux to be fully upwind; the trapping function at the three faces; the faces to the
    horizon; the mass inside the face and the flux through it; the face's radius in units of the horizon mass;
    and the margin in physical units, eq:exc:marginphys; and `fold`, the fold monitor's evaluation (`fold_monitor`),
    formed once the assertions hold, which the driver asserts (`check_fold`) after the step's row is written.

    Raises:
        ExcisionError: If either assertion fails, the run ends, since no admissible move is inward; so it does if the
            fold monitor meets a value that is not finite.
    """
    j_e, N = layout.j_e, layout.N
    mu = outflow_margin(j_e, state, d, geo, eos, layout)
    if not mu > 0.0:
        raise ExcisionError("the outflow margin mu > 0", j_e, mu)
    h = tuple(float(report.h[j]) for j in (j_e, min(j_e + 1, N), min(j_e + 2, N)))
    for offset, value in enumerate(h):
        if not value < 0.0:
            raise ExcisionError("the faces j_e .. j_e + 2 trapped", j_e + offset, value)
    Theta, a = float(speeds.Theta[j_e]), float(speeds.a[j_e])
    alpha = eos.alpha_float
    # the physical margin: (H R_H e^(alpha xi) / alpha) mu, with H R_H = e^(-xi) in the units of the paper
    physical = math.exp((alpha - 1.0) * xi) / alpha * mu
    apparent = report.apparent
    return FaceValues(
        j_e=j_e,
        mu=mu,
        sound_margin=-(Theta + a),
        a_over_Theta=a / abs(Theta) if Theta != 0.0 else float("inf"),
        Lambda_plus=max(Theta + a, 0.0),
        h=(h[0], h[1], h[2]),
        faces_to_horizon=(apparent.j - j_e) if apparent is not None else -1,
        M_e=state.M_e,
        F_e=F_e,
        R_e_over_M_AH=float(geo.X[j_e]) * math.exp(alpha * xi) / report.M_AH if apparent is not None else float("nan"),
        physical_margin=physical,
        fold=fold_monitor(state, d, geo, eos, layout, report.M_AH, xi),
    )


# --- the fold monitor ---


def taub_velocity(r: float, w: float) -> float:
    """The relative velocity `|v_12| = tanh(vartheta)` across a shock of compression `r = rho_2 / rho_1`, `P = w rho`.

    The Taub relation eq:eul:taubeta, `sinh(vartheta) = sqrt(w) (r - 1) / ((1 + w) sqrt(r))`, for any `w`; `tanh`
    from `sinh` as `s / sqrt(1 + s^2)`.
    """
    s = math.sqrt(w) * (r - 1.0) / ((1.0 + w) * math.sqrt(r))
    return s / math.sqrt(1.0 + s * s)


def fold_monitor(
    state: State, d: Derived, geo: Geometry, eos: EquationOfState, layout: Layout, M_AH: float, xi: float
) -> FoldValues:
    """The fold monitor of Section 8.3 on an excised slice, out to the label radius of `R = 2 M_AH`.

    The zone's radius is `X = 2 M_AH e^(-alpha xi)`, as `horizon.near_zone` forms its radii; `fold_numbers` does the
    rest.
    """
    X_zone = FOLD_ZONE * M_AH * math.exp(-eos.alpha_float * xi)
    return fold_numbers(d.rho, state.U, d.Gammabar2, geo, layout, X_zone, eos.w_float)


def fold_numbers(
    rho: FloatArray,
    U: FloatArray,
    Gammabar2: FloatArray,
    geo: Geometry,
    layout: Layout,
    X_zone: float,
    w: float,
) -> FoldValues:
    """The fold monitor's numbers (Section 8.3, eq:eul:boost, eq:eul:taubeta).

    The zone is the retained cells `c` from `j_e + 1` whose inner face lies inside `X_zone`, each with a neighbour to
    either side. A front is detected by the jump `rho_(c-1) / rho_(c+1)` across a cell, above `FOLD_DETECTION`: a
    ratio of cell densities two cells apart, not a face's reconstruction, which sees only a fraction of a front
    captured over two to three cells (a compression of 6 shows face jumps of 1.2, and two-cell jumps of 3.8). Only one
    orientation is sought, the denser side inside: a fold needs an outward-facing shock, whose unshocked side is the
    outer one (`v_12 > 0` with `Utilde_1 < 0`); an inward-facing shock only raises `Gammabar`.

    A front is a run `a..b` of adjacent detected cells. Its compression is measured across it, `rho_2 / rho_1` with
    `rho_2` the densest of the cells `a - FOLD_SPAN .. b` and `rho_1` the least dense of `a .. b + FOLD_SPAN`, which
    reaches the plateaux to either side of a captured front. The pre-shock state is read at the outer face of cell
    `b + 1`, the first face clear of the detected cells, as near the front as its capture allows: what remains of the
    smear there lowers `Gammabar / |Utilde|`, which falls through an outward-facing shock in a trapped region
    (eq:eul:boost), so the reading errs toward an abort. With no front detected, the cell of the largest jump is
    measured as a front `c..c`, so that the row says how near the slice came; only a detected front can abort.

    Under-resolution reads as a front: the smooth infall of Michel's solution itself exceeds the detection level at
    `R = M` once a cell there is wider than about `0.12 M`, and is then measured across as though it were one.

    Raises:
        ExcisionError: If the zone's radius or a density in the zone is not finite.
    """
    N, j_e = layout.N, layout.j_e
    if not math.isfinite(X_zone):
        raise ExcisionError(FOLD_CHECK, j_e, X_zone, "the radius 2 M_AH is not finite")
    last = min(int(geo.X.searchsorted(X_zone, side="right")) - 1, N - 2)  # the zone's last cell
    if last <= j_e:
        return FoldValues(X_zone, 0, math.nan, -1, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
    jumps = rho[j_e:last] / rho[j_e + 2 : last + 2]  # across the cells j_e + 1 .. last
    k = int(jumps.argmax())  # the first NaN if there is one
    jump = float(jumps[k])
    if not math.isfinite(jump):
        raise ExcisionError(FOLD_CHECK, j_e + 1 + k, jump, "a density in the zone is not finite")

    def front(a: int, b: int, c: int, fronts: int) -> FoldValues:
        lo = max(a - FOLD_SPAN, j_e)
        cells: list[float] = rho[lo : min(b + FOLD_SPAN, N - 1) + 1].tolist()  # a few cells: Python is quicker
        rho_2, rho_1 = max(cells[: b + 1 - lo]), min(cells[a - lo :])
        p = b + 2  # the outer face of cell b + 1; b <= N - 2
        U_1, Gammabar_1 = float(U[p]), math.sqrt(float(Gammabar2[p]))  # Gammabar^2 > 0 on an accepted state
        threshold = Gammabar_1 / -U_1 if U_1 < 0.0 else math.inf
        v12 = taub_velocity(rho_2 / rho_1, w)
        X_c = float(geo.Xm[c])
        return FoldValues(X_zone, fronts, jump, c, X_c, rho_2 / rho_1, U_1, Gammabar_1, v12, threshold, v12 / threshold)

    if not jump > FOLD_DETECTION:  # the usual case, at every step without a front
        return front(j_e + 1 + k, j_e + 1 + k, j_e + 1 + k, 0)
    detected = np.flatnonzero(jumps > FOLD_DETECTION)
    breaks = np.flatnonzero(np.diff(detected) > 1)  # the runs of adjacent detected cells
    starts, ends = np.concatenate(([0], breaks + 1)), np.concatenate((breaks, [detected.size - 1]))
    worst: FoldValues | None = None
    for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
        run = detected[s : e + 1]
        peak = int(run[np.argmax(jumps[run])])
        fold = front(j_e + 1 + int(run[0]), j_e + 1 + int(run[-1]), j_e + 1 + peak, len(starts))
        if worst is None or fold.ratio > worst.ratio:
            worst = fold
    assert worst is not None
    return worst


def check_fold(fold: FoldValues) -> None:
    """Abort if the slice folds behind a detected front: `v_12 >= Gammabar_1 / |Utilde_1|` (Section 8.3).

    Behind such a front `Gammabar` of eq:eul:boost is negative, the areal radius decreases outward and the fields are
    two-valued in it: the run has left the formulation, not failed in its arithmetic. The cure is to excise further
    out, toward `2 M`, where no subluminal boost reaches the threshold. Equality counts as a fold: there `Gammabar_2`
    vanishes, and the fold begins.

    Raises:
        ExcisionError: On a violation, with `face` the front's cell and `value` the ratio `v_12 / threshold`.
    """
    if fold.fronts > 0 and not fold.ratio < 1.0:
        raise ExcisionError(
            FOLD_CHECK,
            fold.cell,
            fold.ratio,
            f"the slice folds behind a shock of compression {fold.compression:.4g} at X = {fold.X:.4g} (v_12 = "
            f"{fold.v12:.4g} against {fold.threshold:.4g}): the areal radius decreases outward there and the run has "
            "left the formulation; excise further out, toward 2 M_AH (raise excision.eta and eta_r)",
        )


# --- moving the face outward ---


def re_excision_face(report: HorizonReport, layout: Layout, eta_r: float) -> int | None:
    """The face to advance to by re-excision, or `None` if the face stays (Section 8.3).

    `ceil(N eta_r x_AH)` if it exceeds `j_e`, or as far as leaves the three faces from the new face trapped; with no
    ramp factor, the map being pinned by then.
    """
    apparent = report.apparent
    if apparent is None:
        return None
    N, j_e = layout.N, layout.j_e
    target = math.ceil(N * eta_r * apparent.x)
    if target <= j_e:
        return None
    h = report.h
    for j in range(min(target, N - 3), j_e, -1):
        if h[j] < 0.0 and h[j + 1] < 0.0 and h[j + 2] < 0.0:
            return j
    return None


def zone_needs_extension(report: HorizonReport, zones: tuple[Zone, ...], fraction: float) -> bool:
    """Whether the apparent horizon has come within `fraction` of the outermost zone's transition, or beyond it."""
    apparent = report.apparent
    if apparent is None or not zones:
        return False
    return apparent.x >= fraction * zones[-1].inner_edge


def packed_deviation(
    state: State, geo: Geometry, layout: Layout, dV: FloatArray, whole: BoolArray | None = None
) -> FloatArray:
    """The packed deviation of an excised state: `E - Delta V`, `U - X`, `M_e - X_e^3`, `W`.

    A cell `whole` marks is stored whole (`storage.py`) and keeps its content `E` itself.
    """
    M_e = state.M_e - float(geo.X[layout.j_e]) ** 3  # the mass inside the face against its FRW value
    E = state.E - dV if whole is None else np.where(whole, state.E, state.E - dV)
    return layout.pack(State(E=E, U=state.U - geo.X[: layout.N + 1], W=state.W, M_e=M_e))

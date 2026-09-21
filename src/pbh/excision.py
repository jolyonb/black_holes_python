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
point the closure's rows touch lies where nothing returns; and the map's transition `x_t + Delta_t = (c_t +
c_Delta) x_AH` lies below `0.8`, so that the outer face sits on the static part of the map. If any fails the step
is taken unexcised and the test repeated: the trapped region is a thin shell when it first appears and thickens
inward. A refused attempt is an event, and so is the switch.

Every later step asserts `mu > 0` at the face and the three faces trapped; neither has a cure, since no admissible
move of the face is inward, so a failure ends the run. Outward the face may move at any time: if
`ceil(N eta_r x_AH)` exceeds `j_e`, the face advances to it, or as far as leaves the three faces trapped, and the
mass inside is reset to the cumulative sum at the new face, non-decreasing by construction. The same move takes
the face to a horizon that appears further out, whatever lies between (the multi-scale decision): causality
depends only on the face.

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
from pbh.horizon import FaceValues, HorizonReport
from pbh.layout import Layout
from pbh.maps import Zone
from pbh.state import State
from pbh.types import FloatArray

OUTER_STATIC_LABEL = 0.8
"""The transition must end below this label, so that the outer face sits on the static part of the map."""


class ExcisionError(Exception):
    """An assertion of the excised scheme failed: the run has left the regime the closure is certified in."""

    def __init__(self, check: str, face: int, value: float) -> None:
        super().__init__(f"{check} failed at face {face}: {value!r}")
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
        transition_fits: Test four, `x_t + Delta_t < 0.8`, with the two labels.
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
    alpha = float(eos.alpha)
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
) -> SwitchAttempt:
    """Run the four tests of Section 8.2 on the apparent horizon of `report`; the caller must have one."""
    apparent = report.apparent
    assert apparent is not None, "a switch-on needs an apparent horizon"
    N = layout.N
    alpha = float(eos.alpha)
    x_e = excision.eta * math.exp(-alpha * excision.tau_on) * apparent.x
    j_e = max(math.ceil(N * x_e), layout.j_e)  # a face already further out stays where it is
    inside = 1 <= j_e < apparent.j
    mu = outflow_margin(j_e, state, d, geo, eos, layout) if inside else float("nan")
    h = tuple(float(report.h[j]) if j <= N else float("nan") for j in (j_e, j_e + 1, j_e + 2))
    three = inside and j_e + 2 <= N and all(value < 0.0 for value in h)
    # the transition, sized from the horizon; an extension starts it beyond the last zone's end at the least
    Delta_t = excision.c_Delta * apparent.x
    x_t = excision.c_t * apparent.x
    if zones:
        x_t = max(x_t, zones[-1].outer_edge + Delta_t)
    fits = x_t + Delta_t < OUTER_STATIC_LABEL
    no_overlap = not zones or x_t - Delta_t >= zones[-1].outer_edge - 1e-12 * zones[-1].outer_edge
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
        no_overlap=no_overlap,
    )


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
    """Assert `mu > 0` at the face and the three faces trapped, and collect the face monitors.

    The values (output specification, Section 8): the light-cone margin `mu`; the outflow margin for sound
    `-(Theta + a)`; `a / |Theta|`, below `sqrt(w)` in the certified regime; `Lambda^+ = max(Theta + a, 0)`, which
    must vanish for the flux to be fully upwind; the trapping function at the three faces; the faces to the
    horizon; the mass inside the face and the flux through it; the face's radius in units of the horizon mass;
    and the margin in physical units, eq:exc:marginphys.

    Raises:
        ExcisionError: If either assertion fails; the run ends, since no admissible move is inward.
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
    alpha = float(eos.alpha)
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


def packed_deviation(state: State, geo: Geometry, layout: Layout, dV: FloatArray) -> FloatArray:
    """The packed deviation of an excised state: `E - Delta V`, `U - X`, `M_e - X_e^3`, `W`."""
    M_e = state.M_e - float(geo.X[layout.j_e]) ** 3  # the mass inside the face against its FRW value
    return layout.pack(State(E=state.E - dV, U=state.U - geo.X[: layout.N + 1], W=state.W, M_e=M_e))

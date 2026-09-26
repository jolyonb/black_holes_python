"""The collapse of the core before any horizon forms: the observables of a threshold study, and the bounce.

Near threshold the core approaches the critical solution, which for a perfect fluid is self-similar with its
compactness `2m/R` bounded well below one at every scale. A run just below threshold follows it to small scales and
disperses; one just above leaves it late and forms a small hole. The trapping margin is therefore no continuous
observable of the threshold: its minimum over a run tends to the critical solution's value from below threshold and
jumps to zero above it. What scales continuously, with one exponent `gamma` on both sides, is the mass of the hole
above threshold, `M ~ (C - C_*)^gamma`, which the read-out measures, and below it the peak physical central density,
`rho_max ~ (C_* - C)^(-2 gamma)`. The margin remains the flag of formation.

The physical central density, in units of the background at `xi = 0`, is `rho_phys = rhotilde_0 e^(-2 xi)`, since
the background falls as `H^2 = e^(-2 xi)`. The tilde density alone is measured against a background that is itself
falling, and does not scale. The background's fall also makes the physical density largest at the start in any run
that is not collapsing hard, so the collapse's peak is the largest value after the physical density first rises.

A bounce is positive evidence, never the absence of a horizon by the end of a run: the physical central density must
have fallen to `BOUNCE_FALL` of its peak and stayed there for `BOUNCE_HOLD` in `xi`, and the core margin must by then
have risen above its minimum. A run that ends before either is undecided. A configuration can bounce and collapse
again; the hold is what guards against calling a pause a bounce, and a run that is not stopped at the bounce records
any later formation as usual.

All of this reads the series before the first formation: after it the first retained cell is at the excision face,
not the centre. The resolution of the core at the peak is read from the step record's second-tier monitors (the
cells inside the half-density radius, the viscous against the physical pressure there, the limiter's clipping in
the core), which are written at snapshots, or every step with `monitor_every_step`, as a threshold study should.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from pbh.types import FloatArray

BOUNCE_FALL = 0.5
"""The fraction of its peak to which the physical central density must fall for a bounce ..."""

BOUNCE_HOLD = 0.5
"""... and the time in `xi` for which it must stay there."""


@dataclass(frozen=True)
class CollapseHistory:
    """What the series before formation say about the core.

    Attributes:
        peak_xi: When the physical central density peaked, after it first rose; NaN if it never rose.
        peak_rho_phys: The peak, `rhotilde_0 e^(-2 xi)` in units of the background at `xi = 0`; NaN if none.
        peak_rho_tilde: The tilde central density then.
        margin_min: The smallest core margin of the series, `margin_min_xi` when.
        bounce_xi: When the bounce was established, at the end of the hold; `None` if it was not.
    """

    peak_xi: float
    peak_rho_phys: float
    peak_rho_tilde: float
    margin_min: float
    margin_min_xi: float
    bounce_xi: float | None


def physical_density(xi: FloatArray, rho_0: FloatArray) -> FloatArray:
    """The central density in units of the background at `xi = 0`: `rhotilde_0 e^(-2 xi)`."""
    return rho_0 * np.exp(-2.0 * xi)


def peak_index(rho_phys: FloatArray) -> int | None:
    """The index of the collapse's peak: the largest value after the physical density first rises; `None` if never."""
    rising = np.flatnonzero(np.diff(rho_phys) > 0.0)
    if rising.size == 0:
        return None
    first = int(rising[0])
    return first + int(np.argmax(rho_phys[first:]))


def bounce_time(xi: FloatArray, rho_phys: FloatArray, margin: FloatArray, peak: int) -> float | None:
    """When a bounce after `peak` was established, the end of the first hold that passes; `None` if none has.

    A hold starts where the density falls to `BOUNCE_FALL` of the peak and passes if it stays there through
    `BOUNCE_HOLD` in `xi` and the margin at its end is above the smallest margin before its start.
    """
    fallen = rho_phys <= BOUNCE_FALL * rho_phys[peak]
    fallen[: peak + 1] = False
    starts = np.flatnonzero(fallen & ~np.concatenate(([False], fallen[:-1])))
    for s in starts:
        end = xi[s] + BOUNCE_HOLD
        if xi[-1] < end:
            return None  # the series does not reach the end of this hold yet
        window = (xi >= xi[s]) & (xi <= end)
        e = int(np.flatnonzero(window)[-1])
        if np.all(fallen[window]) and margin[e] > np.min(margin[: s + 1]):
            return float(xi[e])
    return None


def collapse_history(xi: FloatArray, rho_0: FloatArray, margin: FloatArray) -> CollapseHistory:
    """The history of the core from the series `(xi, rhotilde_0, core margin)` before formation, sampled alike."""
    rho_phys = physical_density(xi, rho_0)
    k = int(np.argmin(margin))
    peak = peak_index(rho_phys)
    if peak is None:
        return CollapseHistory(np.nan, np.nan, np.nan, float(margin[k]), float(xi[k]), None)
    return CollapseHistory(
        peak_xi=float(xi[peak]),
        peak_rho_phys=float(rho_phys[peak]),
        peak_rho_tilde=float(rho_0[peak]),
        margin_min=float(margin[k]),
        margin_min_xi=float(xi[k]),
        bounce_xi=bounce_time(xi, rho_phys, margin, peak),
    )


@dataclass
class CoreWatch:
    """The series a run collects before formation, to establish a bounce as it happens (`driver.py`).

    Attributes:
        xi, rho_0, margin: The time, the tilde central density and the core margin of every step so far.
        checked: When the series was last tested.
        bounced: Whether the bounce has been recorded; the watch stops then, or at formation.
    """

    xi: list[float] = field(default_factory=lambda: list[float]())
    rho_0: list[float] = field(default_factory=lambda: list[float]())
    margin: list[float] = field(default_factory=lambda: list[float]())
    checked: float = -math.inf
    bounced: bool = False

    def add(self, xi: float, rho_0: float, margin: float) -> None:
        """Append one step."""
        self.xi.append(xi)
        self.rho_0.append(rho_0)
        self.margin.append(margin)

    def history(self) -> CollapseHistory:
        """The history of the series so far."""
        return collapse_history(np.array(self.xi), np.array(self.rho_0), np.array(self.margin))

"""Reading the black-hole mass (paper Sections 6.3 and 8.5): the rate-corrected estimate and its error bar.

The hole keeps accreting, so the apparent-horizon mass at any finite time is not the mass. With `omega = d ln M/d xi`
the logarithmic growth rate, which a run measures, the estimate

    M_est = M_AH / (1 - omega)                                                                   eq:exc:rateest

equals the final mass identically in time whenever the accretion rate scales as `rho_b M^2` with a constant
efficiency, whatever that efficiency is: it has cancelled, because the rate is measured and not modelled. If the
efficiency drifts, at `varsigma = d ln lambda_a / d xi`, the estimate drifts at `varsigma omega / (1 - omega)`
(eq:exc:estdrift), and the quantity

    Q = ln[omega / (1 - omega)] + xi = ln(lambda_a M_est / R_H),   dQ/d xi = varsigma / (1 - omega),   eq:exc:Q

is constant while the efficiency is. The residual of the estimate is `-int omega dQ` over the future, and the
variation `Q` showed over the preceding e-fold is the proxy for the variation still to come, so the estimate carries
its own error bar,

    delta M / M <~ omega [max Q - min Q] over [xi - 1, xi].                                      eq:exc:bar

The recipe of Section 8.5, which this module carries out on the series `(xi, M_AH)` that the horizon table records
every step:

* `M_AH` is resampled uniformly in `xi`, by linear interpolation of `ln M_AH`, because the steps crowd together after
  formation and an unweighted fit to them would be dominated by wherever the steps were short;
* `omega` is the slope of a straight line fitted to `ln M_AH` over a window of width 0.3 centred on the reading's
  time, a window because `M_AH` comes from an interpolated root and carries steps at the scale of the grid; `M_AH` of
  the reading is the line's value at the centre;
* `Q` and the bar are formed from that `omega`, the bar from the largest and smallest `Q` over the preceding e-fold;
* the mass is read no earlier than two e-folds after the formation of the horizon, and a run is carried until the bar
  falls below the accuracy wanted. The bar looks backward, so only the rate window reaches past the reading's time,
  by half its width.

Also reported with every reading: `lambda_c epsilon = lambda_c M_AH e^(-xi)`, with `lambda_c` the Michel accretion
eigenvalue, and the measured efficiency in its units, `lambda_a / lambda_c = omega / (lambda_c epsilon)`, which
relaxes to one as the near zone becomes the Michel flow.

And the fit's own systematic, which the bar does not contain. A straight line across the curving `ln M_AH` errs in
its slope by `f''' h^2 / 10` and in its value at the centre by `f'' h^2 / 6`, `h` the window's half-width; on steady
accretion `f'' = -omega (1 - omega)` and `f''' = omega (1 - omega)(1 - 2 omega)`, and the two errors, of opposite
sign, leave the estimate low by

    omega h^2 / 15 = omega W^2 / 60                                                              (W the window)

to leading order in `omega`, `1.5e-3 omega` for the production window. The bias shifts the fitted `omega` by a nearly
constant fraction, so it offsets `Q` by a nearly constant amount and the bar, which measures how `Q` varies, cannot
see it. It is `1e-4` or less at the floor up to ten per cent above threshold, far below any bar a run is carried to,
and it is reported with each reading for the error budget rather than corrected: a correction would assume the
steady accretion the estimate exists not to assume, and a higher-order fit would amplify the grid-scale steps of
`M_AH` that the window is there to average.

A series here is one epoch: the stretch after the formation of the horizon being read, with no jump of `M_AH` in it.
A larger trapped region engulfing the hole starts a new epoch (the multi-scale decision), whose floor is counted from
its own formation. Whether it has is read off the finder's report, not off the jump: the apparent horizon's trapped
region is a new one when it has an inner boundary outside the previous apparent horizon, so that it and the old
region are disjoint (`starts_new_epoch`). `Epoch` collects one epoch's series as a run goes.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from pbh.eos import EquationOfState
from pbh.horizon import HorizonReport
from pbh.types import FloatArray


@dataclass(frozen=True)
class ReadoutSettings:
    """The constants of the read-out (Section 8.5).

    Attributes:
        window: The width in `xi` of the straight-line fit that gives `omega`, `0.3`.
        spacing: The spacing in `xi` of the uniform resampling of `M_AH`, and of the readings.
        floor: The e-folds after formation before which no mass is read, `2`.
        bar_span: The e-folds over which the bar takes the variation of `Q`, `1`.
        target: The bar below which the mass is read, the accuracy wanted.
    """

    window: float = 0.3
    spacing: float = 0.005
    floor: float = 2.0
    bar_span: float = 1.0
    target: float = 0.01

    def __post_init__(self) -> None:
        """The window must hold enough resampled points for a fit, and every length must be positive."""
        if min(self.window, self.spacing, self.floor, self.bar_span, self.target) <= 0.0:
            raise ValueError("every read-out setting must be positive")
        if self.window < 4.0 * self.spacing:
            raise ValueError(f"the rate window {self.window} holds too few points at spacing {self.spacing}")


@dataclass(frozen=True)
class Readings:
    """The read-out quantities at uniformly spaced times of one epoch (arrays of equal length).

    Attributes:
        xi: The time of each reading, the centre of its rate window.
        M_AH: The apparent-horizon mass there, the fitted line's value at the centre (units of `R_H`).
        omega: `d ln M_AH / d xi`, the fitted slope.
        M_est: The rate-corrected estimate `M_AH / (1 - omega)`.
        Q: `ln[omega / (1 - omega)] + xi`; NaN where `omega` is not in `(0, 1)`.
        bar: The error bar `omega [max Q - min Q]` over the preceding `bar_span`; NaN until a whole span of readings
            lies behind, and wherever `Q` is NaN in it.
        lambda_c_eps: `lambda_c M_AH e^(-xi)`.
        efficiency: The measured efficiency in units of the Michel value, `omega / (lambda_c epsilon)`.
        systematic: The fit's bias of the estimate, `omega W^2 / 60`, by which it reads low; not in the bar.
    """

    xi: FloatArray
    M_AH: FloatArray
    omega: FloatArray
    M_est: FloatArray
    Q: FloatArray
    bar: FloatArray
    lambda_c_eps: FloatArray
    efficiency: FloatArray
    systematic: FloatArray


def resample(xi: FloatArray, M_AH: FloatArray, spacing: float) -> tuple[FloatArray, FloatArray]:
    """`ln M_AH` interpolated linearly to times uniform in `xi`, from the first sample by `spacing` to the last.

    Args:
        xi: The step times, strictly increasing.
        M_AH: The apparent-horizon mass at those times, positive.
        spacing: The spacing of the uniform times.

    Returns:
        `(times, ln M_AH at them)`.
    """
    if np.any(np.diff(xi) <= 0.0):
        raise ValueError("the times of the series must increase strictly")
    start, span = float(xi[0]), float(xi[-1] - xi[0])
    times: FloatArray = start + spacing * np.arange(math.floor(span / spacing * (1.0 + 1e-12)) + 1, dtype=float)
    return times, np.interp(times, xi, np.log(M_AH))


def readings(xi: FloatArray, M_AH: FloatArray, eos: EquationOfState, settings: ReadoutSettings) -> Readings:
    """Every reading of one epoch: at each resampled time whose whole rate window lies inside the series.

    Args:
        xi: The step times of the epoch.
        M_AH: The apparent-horizon mass at those times.
        eos: The equation of state, for the Michel eigenvalue `lambda_c`.
        settings: The read-out constants.

    Returns:
        The `Readings`, empty if the series is shorter than one window.
    """
    times, ln_M = resample(xi, M_AH, settings.spacing)
    half = round(0.5 * settings.window / settings.spacing)  # samples each side of the centre
    centres = np.arange(half, times.size - half)
    offsets = settings.spacing * np.arange(-half, half + 1)  # the window's times less its centre, symmetric
    windows = ln_M[centres[:, None] + np.arange(-half, half + 1)[None, :]]  # one row per reading
    # the least-squares line through each window: symmetric offsets make the mean offset zero, so the intercept at the
    # centre is the mean and the slope is sum(offset y) / sum(offset^2)
    ln_M_fit = windows.mean(axis=1)
    omega = windows @ offsets / float(offsets @ offsets)
    t = times[centres]
    M = np.exp(ln_M_fit)
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = np.where((omega > 0.0) & (omega < 1.0), np.log(omega / (1.0 - omega)) + t, np.nan)
    lambda_c_eps = eos.accretion_eigenvalue * M * np.exp(-t)
    return Readings(
        xi=t,
        M_AH=M,
        omega=omega,
        M_est=M / (1.0 - omega),
        Q=Q,
        bar=_bars(omega, Q, settings),
        lambda_c_eps=lambda_c_eps,
        efficiency=omega / lambda_c_eps,
        systematic=omega * settings.window**2 / 60.0,
    )


def _bars(omega: FloatArray, Q: FloatArray, settings: ReadoutSettings) -> FloatArray:
    """`omega [max Q - min Q]` over the readings of the preceding `bar_span` (inclusive), NaN until a whole span."""
    span = round(settings.bar_span / settings.spacing)
    bar = np.full(omega.size, np.nan)
    for i in range(span, omega.size):
        behind = Q[i - span : i + 1]
        if np.all(np.isfinite(behind)):
            bar[i] = omega[i] * (float(np.max(behind)) - float(np.min(behind)))
    return bar


def first_reading(r: Readings, xi_formed: float, settings: ReadoutSettings) -> int | None:
    """The index of the first reading that may be quoted: at least `floor` e-folds after `xi_formed`, bar below target.

    Args:
        r: The readings of the epoch.
        xi_formed: The time the epoch's horizon formed.
        settings: The read-out constants.

    Returns:
        The index into `r`, or `None` if no reading qualifies yet.
    """
    eligible = (r.xi >= xi_formed + settings.floor - 1e-9) & (r.bar < settings.target)  # NaN compares false
    hits = np.flatnonzero(eligible)
    return int(hits[0]) if hits.size else None


@dataclass
class Epoch:
    """One epoch's series as a run collects it: the apparent-horizon mass at every step since the epoch began.

    Attributes:
        xi_start: When the epoch's horizon formed; the floor is counted from here.
        xi: The step times, strictly increasing.
        M_AH: The apparent-horizon mass at them.
        X_AH: The apparent horizon's radius at the last step, against which a new trapped region is recognised.
        read: Whether this epoch's mass has been read.
        checked: When the read-out was last tried.
    """

    xi_start: float
    xi: list[float] = field(default_factory=lambda: list[float]())
    M_AH: list[float] = field(default_factory=lambda: list[float]())
    X_AH: float = float("nan")
    read: bool = False
    checked: float = float("-inf")

    def add(self, xi: float, M_AH: float, X_AH: float) -> None:
        """Append a step's apparent horizon; a time not after the last is ignored (a restart re-examines its state)."""
        if not self.xi or xi > self.xi[-1]:
            self.xi.append(xi)
            self.M_AH.append(M_AH)
            self.X_AH = X_AH


def starts_new_epoch(report: HorizonReport, X_AH_previous: float) -> bool:
    """Whether the apparent horizon on this slice bounds a trapped region disjoint from the previous one.

    It does when the sphere just inside it is an inner boundary of a trapped region, at or outside the previous
    apparent horizon: between the two lies an untrapped shell, so the new region is not the old one grown.
    """
    a = report.apparent
    if a is None:
        return False
    inside = [h for h in report.horizons if h.X < a.X]
    return bool(inside) and not inside[-1].outer and inside[-1].X >= X_AH_previous

"""What a finished run says about its black hole, computed from its evolution file (paper Section 8.5).

Nothing here is stored. The run records its raw series, the apparent-horizon mass in the horizon table every step and
the fields in its snapshots, and the one decision it made, the `readout` event; everything below is recomputed from
those with the tools the run used, so it can be recomputed with other settings too. For each epoch of the run (the
stretch after a formation, or after a larger trapped region engulfed the hole):

* the read-out series of `readout.py`, and the reading the run quoted if it quoted one;
* the reference of a long run: the mean of `M_est` over the last half e-fold and its spread there, the
  apparent-horizon mass at the end (a lower bound on the final mass, since it only grows), the efficiency at the end
  and the slope of `Q` over the last e-fold, and the first times after the floor at which the bar fell below five,
  one and three tenths of a per cent;
* the fit of the accretion law eq:exc:ahlaw with a free efficiency over the stretch where `lambda_c epsilon <= 0.05`,
  for comparison with the literature: `1 / M = 1 / M_inf + lambda e^(-xi)` is linear in `e^(-xi)`, so the fit is a
  straight line, and the efficiency is reported in Michel units and as the `F` of Zel'dovich and Novikov,
  `(1 + w) lambda / 4`;
* the enclosed-mass cross-check: the mass inside spheres of fixed physical radius, eq:numbh:mass_obs,
  `m / R_H = (1/2) e^((3 alpha - 2) xi) M_j` interpolated between the faces that bracket the sphere linearly in `X^3`,
  which is exact for the scheme's cells, each of uniform density `rho_c`, from
  every snapshot of the epoch, on a ladder of radii from 3 to 12 final horizon radii `2 M_ref` in steps of 0.25; for
  each sphere its minimum in time, when, `lambda_c epsilon` then, and the leading deficit `54 (M/R)^3` of Section 6.3
  by which the minimum lies below the final mass; and the straight-line extrapolation of the minima in `(M/R)^3` to
  zero, through the spheres whose minimum lies inside the run.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pbh.eos import EquationOfState
from pbh.output import RunReader
from pbh.readout import Readings, ReadoutSettings, first_reading, readings, resample
from pbh.types import FloatArray

LAST = 0.5
"""The e-folds at the end of an epoch over which the reference mass is the mean of the estimate."""

BAR_LEVELS = (0.05, 0.01, 0.003)
"""The bars whose first times after the floor are reported."""

FIT_BELOW = 0.05
"""The fit of the accretion law uses the stretch where `lambda_c epsilon` is at most this ..."""

FIT_SPAN = 1.0
"""... and is made only if that stretch spans at least this many e-folds."""

LADDER = tuple(3.0 + 0.25 * k for k in range(37))
"""The spheres, in final horizon radii: 3 to 12 in steps of 0.25."""


@dataclass(frozen=True)
class Reference:
    """What the end of a long epoch says about its final mass.

    Attributes:
        M_ref: The mean of `M_est` over the last `LAST` e-folds of readings, and `spread` its range there over itself.
        M_AH_end: The apparent-horizon mass at the last step, a lower bound on the final mass.
        efficiency_end: The measured efficiency in Michel units at the last reading.
        Q_slope: The slope of `Q` over the last e-fold of readings.
        bar_below: For each level of `BAR_LEVELS`, the e-folds after formation at which the bar first fell below it
            from the floor on, or `None`.
    """

    M_ref: float
    spread: float
    M_AH_end: float
    efficiency_end: float
    Q_slope: float
    bar_below: dict[float, float | None]


@dataclass(frozen=True)
class AccretionFit:
    """The accretion law fitted with a free efficiency: `M_inf`, the efficiency in Michel units, and `F`."""

    M_inf: float
    efficiency: float
    F: float
    points: int


@dataclass(frozen=True)
class Sphere:
    """The enclosed mass at one sphere of the ladder.

    Attributes:
        k: The radius in final horizon radii, and `R` in units of `R_H`.
        minimum_reached: Whether the smallest mass in time lies inside the epoch rather than at an end.
        m_min: That smallest mass, and `xi_min` when, in e-folds after formation.
        lambda_c_eps: `lambda_c epsilon` then.
        deficit: The leading deficit `54 (M / R)^3` of the plateau below the final mass, as a fraction.
    """

    k: float
    R: float
    minimum_reached: bool
    m_min: float
    xi_min: float
    lambda_c_eps: float
    deficit: float


@dataclass(frozen=True)
class EpochSummary:
    """Everything recomputed about one epoch; `quoted` is the run's `readout` event payload, if it read the mass."""

    xi_start: float
    readings: Readings
    quoted: dict[str, Any] | None
    reference: Reference | None
    fit: AccretionFit | None
    spheres: tuple[Sphere, ...]
    extrapolated: float | None


def epoch_series(reader: RunReader) -> list[tuple[float, FloatArray, FloatArray]]:
    """Each epoch's start and its series `(xi, M_AH)`, from the horizon table split at `formation` and `epoch`."""
    starts = [e.xi for e in reader.events if e.kind in ("formation", "epoch")]
    h = reader.horizon
    xi, M = np.asarray(h["xi"], dtype=float), np.asarray(h["M_AH"], dtype=float)
    series: list[tuple[float, FloatArray, FloatArray]] = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else math.inf
        keep = (xi >= start) & (xi < end) & np.isfinite(M)
        series.append((start, xi[keep], M[keep]))
    return series


def reference(r: Readings, xi_start: float, M_AH_end: float, settings: ReadoutSettings) -> Reference | None:
    """The long-run reference of an epoch's readings; `None` if they span less than one e-fold."""
    if r.xi.size == 0 or r.xi[-1] - r.xi[0] < 1.0:
        return None
    last = r.xi >= r.xi[-1] - LAST
    M_ref = float(np.mean(r.M_est[last]))
    final_efold = (r.xi >= r.xi[-1] - 1.0) & np.isfinite(r.Q)
    below: dict[float, float | None] = {}
    for level in BAR_LEVELS:
        at = ReadoutSettings(settings.window, settings.spacing, settings.floor, settings.bar_span, level)
        i = first_reading(r, xi_start, at)
        below[level] = None if i is None else float(r.xi[i]) - xi_start
    return Reference(
        M_ref=M_ref,
        spread=float(np.ptp(r.M_est[last])) / M_ref,
        M_AH_end=M_AH_end,
        efficiency_end=float(r.efficiency[-1]),
        Q_slope=slope(r.xi[final_efold], r.Q[final_efold]),
        bar_below=below,
    )


def slope(x: FloatArray, y: FloatArray) -> float:
    """The slope of the least-squares line through the points; NaN through fewer than three."""
    return float(np.polyfit(x, y, 1)[0]) if x.size >= 3 else math.nan


def fit_accretion(
    xi: FloatArray, M_AH: FloatArray, eos: EquationOfState, settings: ReadoutSettings
) -> AccretionFit | None:
    """The accretion law with a free efficiency, fitted where `lambda_c epsilon <= FIT_BELOW`, if that spans enough.

    `None` if the stretch spans less than `FIT_SPAN`. `1 / M = 1 / M_inf + lambda e^(-xi)` is a straight line in
    `e^(-xi)`, fitted to the series resampled as the read-out resamples it.
    """
    times, ln_M = resample(xi, M_AH, settings.spacing)
    M = np.exp(ln_M)
    late = eos.accretion_eigenvalue * M * np.exp(-times) <= FIT_BELOW
    if not late.any() or times[late][-1] - times[late][0] < FIT_SPAN:
        return None
    slope, intercept = np.polyfit(np.exp(-times[late]), 1.0 / M[late], 1)
    lam = float(slope)
    return AccretionFit(
        M_inf=1.0 / float(intercept),
        efficiency=lam / eos.accretion_eigenvalue,
        F=(1.0 + float(eos.w)) * lam / 4.0,
        points=int(late.sum()),
    )


def enclosed_mass(reader: RunReader, index: int, R: FloatArray, eos: EquationOfState) -> tuple[float, FloatArray]:
    """The mass inside the spheres of physical radii `R` at one snapshot, eq:numbh:mass_obs; NaN off the grid.

    Returns the snapshot's time and the masses in units of `R_H`.
    """
    record = reader.snapshot(index)
    state = record.state
    alpha = float(eos.alpha)
    X = record.X
    j_e = record.j_e
    M = np.full(X.size, np.nan)
    M[j_e] = state.M_e
    M[j_e + 1 :] = state.M_e + 3.0 * np.cumsum(state.E[j_e:])
    labels = R * math.exp(-alpha * record.xi)  # the label radius of a fixed physical radius at this time
    inside = (labels >= X[j_e]) & (labels <= X[-1])
    m = np.full(R.size, np.nan)
    # within a cell the density is uniform, so the enclosed mass is linear in the volume, X^3
    M_at = np.interp(labels[inside] ** 3, X[j_e:] ** 3, M[j_e:])
    m[inside] = 0.5 * math.exp((3.0 * alpha - 2.0) * record.xi) * M_at
    return record.xi, m


def spheres(
    reader: RunReader,
    start: float,
    end: float,
    M_ref: float,
    series: tuple[FloatArray, FloatArray],
    eos: EquationOfState,
) -> tuple[tuple[Sphere, ...], float | None]:
    """The ladder of spheres over the snapshots in `[start, end)`, and the extrapolation of their minima."""
    k = np.array(LADDER)
    R = 2.0 * M_ref * k
    times: list[float] = []
    masses: list[FloatArray] = []
    for info in reader.snapshots:
        if start <= info.xi < end:
            t, m = enclosed_mass(reader, info.index, R, eos)
            times.append(t)
            masses.append(m)
    if len(times) < 3:
        return (), None
    t_arr, m_arr = np.array(times), np.array(masses)
    xi_series, M_series = series
    out: list[Sphere] = []
    for n in range(k.size):
        column = m_arr[:, n]
        ok = np.isfinite(column)
        if ok.sum() < 3:
            continue
        i = int(np.argmin(np.where(ok, column, np.inf)))
        valid = np.flatnonzero(ok)
        t_min = float(t_arr[i])
        M_then = float(np.interp(t_min, xi_series, M_series))
        out.append(
            Sphere(
                k=float(k[n]),
                R=float(R[n]),
                minimum_reached=bool(valid[0] < i < valid[-1]),
                m_min=float(column[i]),
                xi_min=t_min - start,
                lambda_c_eps=eos.accretion_eigenvalue * M_then * math.exp(-t_min),
                deficit=54.0 / (2.0 * float(k[n])) ** 3,
            )
        )
    reached = [s for s in out if s.minimum_reached]
    extrapolated = None
    if len(reached) >= 3:
        slope_and_intercept = np.polyfit([1.0 / (2.0 * s.k) ** 3 for s in reached], [s.m_min for s in reached], 1)
        extrapolated = float(slope_and_intercept[1])
    return tuple(out), extrapolated


def summarise(reader: RunReader) -> list[EpochSummary]:
    """Every epoch of the run, summarised; an empty list if no horizon formed."""
    config = reader.config
    eos = config.fluid.build()
    settings = config.readout.build()
    quoted = [e for e in reader.events if e.kind == "readout"]
    out: list[EpochSummary] = []
    epochs = epoch_series(reader)
    for n, (start, xi, M_AH) in enumerate(epochs):
        end = epochs[n + 1][0] if n + 1 < len(epochs) else math.inf
        r = readings(xi, M_AH, eos, settings)
        ref = reference(r, start, float(M_AH[-1]), settings)
        fit = fit_accretion(xi, M_AH, eos, settings)
        ladder, extrapolated = spheres(reader, start, end, ref.M_ref, (xi, M_AH), eos) if ref else ((), None)
        own = [e.payload for e in quoted if e.payload["epoch_start"] == start]
        out.append(EpochSummary(start, r, own[0] if own else None, ref, fit, ladder, extrapolated))
    return out


def as_json(summaries: list[EpochSummary]) -> list[dict[str, Any]]:
    """The summaries as plain data, the series included, for export and plotting."""
    result: list[dict[str, Any]] = []
    for s in summaries:
        r = s.readings
        series = {name: getattr(r, name).tolist() for name in r.__dataclass_fields__}
        result.append(
            {
                "xi_start": s.xi_start,
                "quoted": s.quoted,
                "reference": None if s.reference is None else reference_json(s.reference),
                "fit": None if s.fit is None else s.fit.__dict__,
                "spheres": [sphere.__dict__ for sphere in s.spheres],
                "extrapolated": s.extrapolated,
                "series": series,
            }
        )
    return result


def reference_json(ref: Reference) -> dict[str, Any]:
    """The reference as plain data; the bar levels become strings, as JSON keys must."""
    return {**ref.__dict__, "bar_below": {str(level): v for level, v in ref.bar_below.items()}}


def describe(summaries: list[EpochSummary]) -> str:
    """A readable account of the summaries."""
    if not summaries:
        return "no horizon formed"
    lines: list[str] = []
    for n, s in enumerate(summaries):
        lines.append(f"epoch {n}: formed at xi = {s.xi_start:.4f}")
        if s.quoted is not None:
            q = s.quoted
            flag = "  FLAGGED: efficiency far from Michel's" if q["efficiency_flag"] else ""
            lines.append(
                f"  quoted: M_est = {q['M_est']:.5g} R_H +- {100 * q['bar']:.2f}% at xi = {q['xi_reading']:.3f} "
                f"({q['xi_reading'] - s.xi_start:.2f} e-folds), lambda_c eps = {q['lambda_c_eps']:.3f}, "
                f"efficiency {q['efficiency']:.3f} of Michel's{flag}"
            )
        else:
            lines.append("  no reading quoted")
        ref = s.reference
        if ref is not None:
            below = ", ".join(f"{100 * k:g}%: {'-' if v is None else f'+{v:.2f}'}" for k, v in ref.bar_below.items())
            lines.append(
                f"  reference: M_ref = {ref.M_ref:.5g} (spread {ref.spread:.1e} over the last {LAST} e-folds), "
                f"M_AH at end {ref.M_AH_end:.5g}, efficiency at end {ref.efficiency_end:.3f}, "
                f"Q slope {ref.Q_slope:+.3f}"
            )
            lines.append(f"  bar first below (e-folds after formation): {below}")
        if s.fit is not None:
            f = s.fit
            lines.append(
                f"  accretion law, free efficiency: M_inf = {f.M_inf:.5g}, efficiency {f.efficiency:.3f} of Michel's, "
                f"F = {f.F:.3f} ({f.points} points)"
            )
        reached = [sphere for sphere in s.spheres if sphere.minimum_reached]
        if s.spheres:
            count = f"{len(s.spheres)} spheres on the grid, {len(reached)} with a minimum in the run"
            lines.append(f"  enclosed mass: {count}")
            for sphere in reached[:: max(1, len(reached) // 6)]:
                lines.append(
                    f"    {sphere.k:5.2f} R_AH: min {sphere.m_min:.5g} at +{sphere.xi_min:.2f}, "
                    f"lambda_c eps {sphere.lambda_c_eps:.3f}, leading deficit {100 * sphere.deficit:.2f}%"
                )
            if s.extrapolated is not None:
                lines.append(f"  minima extrapolated in (M/R)^3 to zero: {s.extrapolated:.5g}")
    return "\n".join(lines)

"""The error of a run from a pair of runs at `N` and `N/2` (Section 8.6, the error budget, Table tab:numbh:errors).

A run's spatial error is measured, not modelled: the same datum is run again on the grid with half the cells
(`pbh run --pair`, about a quarter more work), and since the scheme is second order the error of the run at `N` is
a third of the difference, `(v_N - v_(N/2)) / 3`. Everything here is a pure function of the two runs' summaries
(`summary.py`), computed after the fact; nothing is stored.

* The outcome of each run, formed, bounced, undecided or aborted, and whether the two agree.
* For each observable, the formation time, the quoted `M_est`, the peak physical central density and the bounce time,
  the two values; the mass carries its spatial error with the coverage factor `F_COVERAGE`, the only calibrated one;
  the peak density carries the bare third of the difference, uncalibrated; the two times carry none, for each is
  sampled at the steps and its error is about a step, first order, which the third of a difference does not bound.
* The threshold-trust verdict: is the run far enough from threshold that its resolution error cannot have put it
  on the wrong side? Near threshold the observables scale with one exponent `GAMMA`, `M ~ (C - C_*)^GAMMA` above and
  `rho_max ~ (C_* - C)^(-2 GAMMA)` below, so a shift `dC_*` of the threshold moves them by `dM/M = GAMMA dC_* /
  (C - C_*)` and `d ln rho_max = -2 GAMMA dC_* / (C_* - C)`. The pair measures the left-hand side, `x = |dM/M|`
  above and `x = |d ln rho_max| / 2` below, so `x = GAMMA |dC_*| / |C - C_*|` on both sides and the family's
  constant cancels; the error of the run at `N` is a third of the pair's, so its distance from threshold in units of
  its own threshold error is `3 GAMMA / x`, and it is trusted to lie on the side it says if
  `P = Phi(3 GAMMA / (K_TRUST x))` reaches `TRUST_LEVEL`, `Phi` the standard normal distribution: `P` is a gate
  calibrated at that level, not a calibrated probability elsewhere. The resolution at which it reaches
  `TRUST_LEVEL` follows from `x ~ N^(-2)`: `N sqrt(z K_TRUST x / (3 GAMMA))`, `z` the quantile. Outcomes that differ
  between `N` and `N/2` are untrusted outright, and so is a pair either of whose runs aborted. `K_TRUST` was
  calibrated with the step cap at `1e-7`; at a looser cap the threshold moves at `N` and `N/2` alike, which the pair
  cannot see and `P` does not contain, and the pair says so.
* The mass error budget, each component separately and their sum in quadrature: the spatial error
  `F_COVERAGE |dM| / 3`, the read-out bar of the run at `N` (eq:exc:bar), the read-out window's bias, and the quoted
  systematics of the table (time step, viscosity, shock floor, outer boundary, initial data). The systematics hold
  for `d = (C - C_*)/C_* >= 0.03` and grow as `GAMMA / d` nearer threshold, which is printed with the budget. The
  bias of an early reading against the final mass, while the efficiency is still relaxing from its overshoot of the
  Michel value, is inside the bar, which is anchored on Michel (eq:exc:bar).

The constants are calibrated by the campaigns of bite 26 (analysis experiments e8-e10, gathered and validated in
experiments/pair_constants) on the Gaussian and flat-topped profiles of radiation on the sinh grid; each names its
source. Those campaigns made their data by the former recipe of Section 5.4, a profile imposed at the start; the
pair is for the data `pbh initial` writes, the growing solution of a seed, at whose start the threshold and the mass
move only at `O(epsilon2^2)` (Section 5.4), and the constants that depended on the former recipe's start say so.
"""

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

from pbh.config import GridConfig, RunConfig
from pbh.summary import RunSummary

GAMMA = 0.3558
"""The critical exponent of radiation, `M ~ (C - C_*)^GAMMA` (Koike, Hara and Adachi, Phys. Rev. Lett. 74, 5170
(1995)). Kept, not measured: the criticality study (experiments/e10_criticality) finds 0.351 +- 0.005 above
threshold and 0.348 +- 0.024 below, 1.0 and 0.3 sigma from it, the errors set by the correction to scaling, and
neither is as precise."""

K_TRUST = 1.27
"""The width of the trust sigmoid in units of the pair's threshold error: the largest over-prediction of the distance
to threshold by the pair, 2.09 at the grid scale 3 (experiments/pair_constants, e10's pairs between 1 and 30 threshold
errors from threshold at the scales 0.15-3; pairs inside their own threshold error, which no verdict can trust, are
left out), divided by the 95 per cent quantile 1.645, so that every one of them is covered; at
the scale 0.3 alone e10 fits 0.95, which covers 3 of the 7 such pairs at scale 3. The thresholds the distances are
measured from are the formation thresholds, between the largest compaction without a formation event and the
smallest with one. Every calibration run had the step cap at `1e-7` (`CALIBRATED_CAP_TOLERANCE`) and data of the
former recipe imposed at `xi_start = -10`, which realise the seed to `O(eps0^2)`, `5.7e-6` for the Gaussian of width
2; the seed's threshold at `N = 800` agrees with theirs to `9.6e-8` in `C`, within the brackets' resolution
(experiments/seed_check), so the width holds for seed data. The scale-3 pairs rest on a threshold at `N = 400`
bracketed by runs that trapped only the first face and never switched on."""

CALIBRATED_CAP_TOLERANCE = 1e-7
"""The step cap's tolerance of the runs that calibrated `K_TRUST`. The production `1e-5` moves the threshold at `N`
and `N/2` alike, by an amount that grows with the super-horizon e-folds and depends on the stretch: on the sinh
scale 0.3 by 2.3e-7 in `C` from `xi_start = -6`, 4.3e-6 from -10 and 8.8e-6 from -14 (experiments/e10_criticality);
on the production scale 3 from -6 by about 4e-6, inferred from the mass (experiments/e9_systematics). That is
comparable with the pair's own threshold error at `N = 1600`; `P` does not contain it, and a looser cap is named in
the verdict."""

SEED_EPSILON2 = 1e-5
"""The largest start tolerance `initial.epsilon2` at which the initial data's systematic is zero: the seed's datum
errs at relative `O(epsilon2^2)`, the same at `N` and `N/2`; a looser one is named in the verdict."""

F_COVERAGE = 1.53
"""The factor on the pair's spatial error of the mass, `F |dM| / 3`, that makes it cover the true error: the 95 per
cent coverage factor of the 23 pairs of both families with `|dM/M| > 1e-4` (experiments/e8_mass_errors, 1.525;
1.29 from the Gaussian alone). Below `1e-4` the step cap's floor, `SYSTEMATIC_TIME_STEP`, sets the error."""

SYSTEMATIC_TIME_STEP = 1.1e-4
"""The relative error of the mass from the time step, kept as a conservative bound: measured with the step cap at
its former default tolerance `1e-5` against `1e-7` (the default since 2026-10-02, at which the cap's part vanishes;
to be remeasured with the campaign rerun at `1e-7`, which also recalibrates `F_COVERAGE`),
1.03e-4 at 0.03 above threshold, rounded up (experiments/e9_systematics); the Courant number alone, 0.75 against
0.375, 4e-7. Measured at `xi_start = -6` on the sinh scale 3: it grows as `1/(C - C_*)` nearer threshold, it grows
with an earlier start, since the cap's amplitude error accumulates over the super-horizon steps (the threshold moves
by 4.3e-6 in `C` from -10 and 8.8e-6 from -14 on the scale 0.3, experiments/e10_criticality), and it vanishes with a
tighter cap."""

SYSTEMATIC_VISCOSITY = 0.0
"""The relative error of the mass from the viscosity `c_v`: zero, for `c_v` from 0.5 to 1.5 moves it by 4.3e-5 and
1.1e-5 at `N` = 800 and 1600 (experiments/e9_systematics), a second-order error the pair already measures."""

SYSTEMATIC_SHOCK_FLOOR = 0.0
"""The relative error of the mass from the shock floor: zero, for no front crosses an excised step at `N >= 800`
from 0.03 to 0.3 above threshold (experiments/e9_systematics)."""

SYSTEMATIC_OUTER_BOUNDARY = 0.0
"""The relative error of the mass from the outer boundary: zero, for moving it from `Rtilde_max` 30 to 45 changes the
mass by 6.3e-7 (experiments/e9_systematics)."""

SYSTEMATIC_INITIAL_DATA = 0.0
"""The relative error of the mass from the initial data: zero for a seed started at `initial.epsilon2 <= 1e-5`
(`SEED_EPSILON2`). The datum is the growing solution to relative `O(eps0^4)`: against the code's own evolution of the
seed from `eps0^2 = 1e-7` it errs by `6.1e-5` of the mass deviation at `eps^2 = 4e-2` and its velocity falls as
`eps^4` (experiments/seed_check), so about `1e-11` at `1e-5`. The former recipe's `9.76e-5` at 0.03 above threshold,
a profile imposed at `xi_start` -6 instead of -10 (experiments/e9_systematics), is an error of that recipe's start,
which the seed does not have."""

SYSTEMATICS = {
    "time_step": SYSTEMATIC_TIME_STEP,
    "viscosity": SYSTEMATIC_VISCOSITY,
    "shock_floor": SYSTEMATIC_SHOCK_FLOOR,
    "outer_boundary": SYSTEMATIC_OUTER_BOUNDARY,
    "initial_data": SYSTEMATIC_INITIAL_DATA,
}
"""The quoted systematics of the mass, as fractions of it, by name; measured at `d = 0.03` and valid for `d >= 0.03`."""

BUDGET_CAVEAT = "not covered: the systematics grow as gamma/d below d = 0.03"
"""What the printed total of the mass budget does not contain (Section 8.6, Table tab:numbh:errors)."""

TRUST_LEVEL = 0.95
"""The probability at or above which a run is trusted to lie on the side of threshold it says."""

OBSERVABLES = ("formation_xi", "M_est", "rho_max", "bounce_xi")
"""The observables compared across the pair: the first formation time, the quoted mass, the first peak of the
physical central density before formation (`collapse.peak_index`, the peak e10 calibrated `K_TRUST` below threshold
with, not a later re-collapse onto an emptied centre) and the time the bounce was established."""

TIMES = ("formation_xi", "bounce_xi")
"""The observables sampled at the steps, whose error is about a step and is not estimated from the pair."""


def half_name(name: str) -> str:
    """The name of the half-resolution companion of the run `name`."""
    return f"{name}.half"


def half_config(config: RunConfig) -> RunConfig:
    """The configuration of the companion: the same with `grid.N` halved and nothing else changed; `N` must be even."""
    N = config.grid.N
    if N % 2 or N < 4:
        raise ValueError(f"a pair needs an even grid.N of at least 4 to halve, not {N}")
    grid = GridConfig.model_validate({**config.grid.model_dump(), "N": N // 2})
    return config.model_copy(update={"grid": grid})


def outcome(s: RunSummary) -> str:
    """How a finished run ended: `aborted` unless it completed, else `formed`, `bounced` or `undecided`."""
    if s.status != "completed":
        return "aborted"
    if s.core is None:  # no step before formation: a run continued from after it
        return "formed" if s.epochs else "undecided"
    return s.core.outcome


def observables(s: RunSummary) -> dict[str, float]:
    """The observables of `OBSERVABLES` of one run; NaN where the run has none."""
    history = None if s.core is None else s.core.history
    quoted = s.epochs[-1].quoted if s.epochs else None  # the last epoch's hole has engulfed any earlier
    bounce = None if history is None else history.bounce_xi
    return {
        "formation_xi": s.epochs[0].xi_start if s.epochs else math.nan,
        "M_est": math.nan if quoted is None else float(quoted["M_est"]),
        "rho_max": math.nan if history is None else history.peak_rho_phys,
        "bounce_xi": math.nan if bounce is None else bounce,
    }


@dataclass(frozen=True)
class Estimate:
    """One observable across the pair: its value at `N`, at `N/2`, and the spatial error of the value at `N`.

    The error is NaN for the times of `TIMES`, which the pair does not estimate.
    """

    name: str
    value: float
    half: float
    error: float


def estimates(s: RunSummary, s_half: RunSummary) -> tuple[Estimate, ...]:
    """Each observable's two values and error `(v_N - v_(N/2)) / 3`, times `F_COVERAGE` for the mass.

    The error is NaN where either value is absent, and for the times of `TIMES`.
    """
    full, half = observables(s), observables(s_half)
    return tuple(
        Estimate(
            name,
            full[name],
            half[name],
            math.nan if name in TIMES else (F_COVERAGE if name == "M_est" else 1.0) * (full[name] - half[name]) / 3.0,
        )
        for name in OBSERVABLES
    )


@dataclass(frozen=True)
class Trust:
    """The threshold-trust verdict of a pair.

    Attributes:
        verdict: `trusted` (probability at least `TRUST_LEVEL`), `untrusted`, or `no verdict` when the runs agree but
            neither the mass of a hole nor the peak density of a bounce is there to measure the distance.
        side: `above` or `below` threshold, or `None` when the outcomes differ, either aborted, or there is no verdict.
        x: The pair's measure of closeness, `|dM/M|` above and `|d ln rho_max| / 2` below; NaN if none.
        probability: `Phi(3 GAMMA / (K_TRUST x))`, `0` when the outcomes differ or either aborted; `None` with no
            verdict.
        N_needed: The even `N` at which the probability would reach `TRUST_LEVEL`; `None` unless measured.
    """

    verdict: str
    side: str | None
    x: float
    probability: float | None
    N_needed: int | None


def normal_cdf(z: float) -> float:
    """The standard normal distribution function, `Phi(z)`."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def trust_from(x: float, N: int, side: str) -> Trust:
    """The verdict on one side of threshold from the pair's closeness `x` at the resolution `N`."""
    probability = normal_cdf(3.0 * GAMMA / (K_TRUST * x)) if x > 0.0 else 1.0
    z = NormalDist().inv_cdf(TRUST_LEVEL)
    N_needed = 2 * max(1, math.ceil(N * math.sqrt(z * K_TRUST * x / (3.0 * GAMMA)) / 2.0))
    verdict = "trusted" if probability >= TRUST_LEVEL else "untrusted"
    return Trust(verdict, side, x, probability, N_needed)


def trust(s: RunSummary, s_half: RunSummary, N: int) -> Trust:
    """The threshold-trust verdict of the pair whose finer run has `N` cells (see the module docstring)."""
    full, half = outcome(s), outcome(s_half)
    if full != half or full == "aborted":
        return Trust("untrusted", None, math.nan, 0.0, None)
    v, v_half = observables(s), observables(s_half)
    if full == "formed" and math.isfinite(v["M_est"] - v_half["M_est"]):
        return trust_from(abs(v["M_est"] - v_half["M_est"]) / v["M_est"], N, "above")
    if full == "bounced" and math.isfinite(v["rho_max"] - v_half["rho_max"]):
        return trust_from(abs(math.log(v["rho_max"] / v_half["rho_max"])) / 2.0, N, "below")
    return Trust("no verdict", None, math.nan, None, None)


@dataclass(frozen=True)
class MassBudget:
    """The error budget of the quoted mass of the run at `N`, in units of `R_H`.

    Attributes:
        M_est: The mass quoted by the run at `N`.
        spatial: `F_COVERAGE |dM| / 3` from the pair.
        readout: The read-out bar of eq:exc:bar times the mass.
        window: The read-out window's bias, `omega W^2 / 60` times the mass, which the bar does not contain: the
            fit's own bias, not the reading's against the final mass, which the bar does contain.
        systematics: Each quoted systematic of `SYSTEMATICS` times the mass, by name.
        total: All of them in quadrature.
    """

    M_est: float
    spatial: float
    readout: float
    window: float
    systematics: dict[str, float]
    total: float


def mass_budget(s: RunSummary, s_half: RunSummary) -> MassBudget | None:
    """The error budget of the quoted mass; `None` unless both runs quoted one."""
    quoted = s.epochs[-1].quoted if s.epochs else None
    dM = observables(s)["M_est"] - observables(s_half)["M_est"]
    if quoted is None or not math.isfinite(dM):
        return None
    M = float(quoted["M_est"])
    spatial = F_COVERAGE * abs(dM) / 3.0
    readout = float(quoted["bar"]) * M
    window = float(quoted["systematic"]) * M
    systematics = {name: fraction * M for name, fraction in SYSTEMATICS.items()}
    total = math.sqrt(spatial**2 + readout**2 + window**2 + sum(v**2 for v in systematics.values()))
    return MassBudget(M, spatial, readout, window, systematics, total)


def caveats(config: RunConfig) -> tuple[str, ...]:
    """The settings of the run at which `K_TRUST`'s calibration or a zero systematic does not hold, said in words."""
    found: list[str] = []
    tolerance = config.stepping.cap_tolerance
    if tolerance > CALIBRATED_CAP_TOLERANCE:
        found.append(
            f"the step cap's tolerance {tolerance:g} is looser than the {CALIBRATED_CAP_TOLERANCE:g} of K's "
            "calibration: it moves the threshold at N and N/2 alike (by up to about 9e-6 in C, more for an earlier "
            "start), which P does not contain"
        )
    epsilon2 = config.initial.epsilon2
    if epsilon2 > SEED_EPSILON2:
        found.append(
            f"initial.epsilon2 = {epsilon2:g} is looser than {SEED_EPSILON2:g}: the seed's datum errs at relative "
            "O(epsilon2^2), at N and N/2 alike, which neither P nor the budget's initial-data term contains"
        )
    return tuple(found)


@dataclass(frozen=True)
class PairSummary:
    """Everything the pair says: the outcomes, the observables with their errors, the verdict, the mass budget.

    `caveats` names the settings of the run at which the verdict's calibration or a zero systematic does not hold.
    """

    N: int
    outcomes: tuple[str, str]
    agree: bool
    estimates: tuple[Estimate, ...]
    trust: Trust
    budget: MassBudget | None
    caveats: tuple[str, ...]


def analyse(s: RunSummary, s_half: RunSummary, config: RunConfig) -> PairSummary:
    """The pair of the run with configuration `config`, summarised as `s`, and its companion, summarised as `s_half`.

    The outcomes agree only if they are equal and neither is an abort, which is no outcome.
    """
    N = config.grid.N
    outcomes = (outcome(s), outcome(s_half))
    agree = outcomes[0] == outcomes[1] != "aborted"
    return PairSummary(
        N, outcomes, agree, estimates(s, s_half), trust(s, s_half, N), mass_budget(s, s_half), caveats(config)
    )


def as_json(pair: PairSummary) -> dict[str, Any]:
    """The pair as plain data, for export."""
    budget = None if pair.budget is None else pair.budget.__dict__
    return {
        "N": pair.N,
        "outcomes": list(pair.outcomes),
        "agree": pair.agree,
        "estimates": {e.name: {"value": e.value, "half": e.half, "error": e.error} for e in pair.estimates},
        "trust": pair.trust.__dict__,
        "budget": budget,
        "caveats": list(pair.caveats),
    }


def describe(pair: PairSummary) -> list[str]:
    """A readable account of the pair."""
    full, half = pair.outcomes
    aborted = [label for label, o in (("N", full), ("N/2", half)) if o == "aborted"]
    agreement = "both aborted" if len(aborted) == 2 else "agree" if pair.agree else "DIFFER"
    lines = [f"pair: N = {pair.N} {full}, N/2 = {pair.N // 2} {half} ({agreement})"]
    t = pair.trust
    if t.probability is None:
        lines.append("  threshold: no verdict (neither a read mass nor a bounce to measure the distance)")
    elif aborted:
        lines.append(f"  threshold: untrusted, the run at {' and the run at '.join(aborted)} aborted")
    elif t.side is None:
        lines.append("  threshold: untrusted, the outcomes differ between N and N/2")
    else:
        lines.append(
            f"  threshold: {t.verdict} {t.side}, P = {t.probability:.3f} (x = {t.x:.3g}); "
            f"{100 * TRUST_LEVEL:g}% needs N >= {t.N_needed}"
        )
    lines += [f"    caveat: {caveat}" for caveat in pair.caveats]
    for e in pair.estimates:
        if not (math.isfinite(e.value) or math.isfinite(e.half)):
            continue
        values = f"  {e.name}: {e.value:.6g} (N/2: {e.half:.6g})"
        if e.name in TIMES:
            lines.append(f"{values}, sampled at the steps: no error estimate")
        elif e.name == "M_est":
            lines.append(f"{values}, spatial error {e.error:+.3g} (F = {F_COVERAGE:g})")
        else:
            lines.append(f"{values}, spatial error {e.error:+.3g} (uncalibrated)")
    b = pair.budget
    if b is not None:
        parts = {"spatial": b.spatial, "readout": b.readout, "window": b.window, **b.systematics}
        breakdown = ", ".join(f"{name} {value:.2g}" for name, value in parts.items())
        lines.append(f"  M_est = {b.M_est:.6g} +- {b.total:.2g} R_H ({breakdown})")
        lines.append(f"    {BUDGET_CAVEAT}")
    return lines

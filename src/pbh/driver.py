"""The driver: one loop that runs a configuration from an initial state to its end, writing the run's files.

A run is `run(config, initial, paths)`. It builds the `Scheme` from the configuration and the initial record, which
carries the map's zones and the formation time if it comes from a snapshot after formation, checks that the record
sits on the grid that gives, saves the configuration with its provenance (with `evolution.xi_start` filled in from the
initial data when the configuration leaves it out, as a fresh run's normally does), and then steps:

1. evaluate the rate at the state (the first stage of the step) and choose the step: the smaller of the Courant
   step and the cap (eq:num:cfl), clipped to land exactly on the next snapshot time and on the end;
2. take the checked Runge-Kutta step in deviation form (Section 7.6): every stage and the result must be finite and
   inside the hyperbolic domain, else the attempt is refused, logged as a `rejection` event, and retried with half
   the step; advance the clock, compensated for steps of a few ulps of `xi` (`timestep.Clock`), and end the run if
   even so it stalls; then move the cells across the storage lines (`storage.py`): a cell below a quarter of the
   background is stored whole from here on, and back as its deviation above a half;
3. record the step's monitors, the full row when the step ends on a snapshot or the configuration asks for it every
   step;
4. examine the state arrived at: run the horizon finder, record formation at the first trapped face, and, with
   excision enabled, throw the switch when its four tests pass, assert the face and form the fold monitor every
   excised step, move the face outward by re-excision, and pin a further zone when the horizon approaches the
   transition; record the horizon row, then assert the fold monitor on it, write the snapshot when due, flush the
   step record on its cadence;
5. read the mass (Section 8.5, `readout.py`): the apparent-horizon mass of every step is collected by epoch, a new
   one beginning when a trapped region appears outside the apparent horizon (an `epoch` event), and every
   `READOUT_CHECK` in `xi` from the floor on the read-out is tried on the epoch's series; the first reading with its
   bar below the target is recorded as a `readout` event, with the near-zone monitors beside the Michel values, the
   outflow margin, the minimum lapse and a flag if the efficiency is not yet near Michel's, and the run ends there
   if the configuration says so;
6. watch the core before formation (`collapse.py`): the central density and the core margin of every step are
   collected, and every `READOUT_CHECK` in `xi` tested for a bounce; one established is recorded as a `bounce` event
   with the peak of the physical central density and the smallest core margin, and the run ends there if the
   configuration says so. A restart hands the watch its series from the source file (`core_watch`).

The initial state is examined the same way before the first step, so that a run restarted from a snapshot makes
the decisions the uninterrupted run made at that state; a restart hands the run its epoch's history of `M_AH` from
the source file (`epoch_history`), so that it reads the mass the uninterrupted run would read, and the core's watch its
series (`core_watch`), so that it records the bounce the uninterrupted run would record. The switch-on writes
a snapshot of the state it is thrown on, unexcised and already carrying the new zone: the same collapse continued
from it with excision off runs on the same grid, which is the comparison Section 8.3 asks for.

An abort is a result, not an exception: if no step the retry policy allows passes the checks (`Gammabar^2 <= 0`, named
by what the accepted state says it is, or the positivity or finiteness of a stage after twenty halvings), the clock
stalls, the outer face is trapped, or an excision assertion fails, the driver records an `abort` event naming what
failed, writes a snapshot of the last good state, and ends the run with the status `aborted`. No density is too small:
a void is followed down by storing its cells whole, and the slicing's own limit shows as one of these aborts. The
far-zone radius of the monitors is read off the initial data. The clock's carry is dropped at every snapshot, so that
a restart from it, which starts with none, continues bit for bit.
"""

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pbh import storage
from pbh.causal import isolation
from pbh.collapse import CoreWatch
from pbh.config import RunConfig, SnapshotChoice, save
from pbh.derived import NotHyperbolicError
from pbh.equations import DerivsResult
from pbh.excision import (
    ExcisionError,
    SwitchAttempt,
    attempt_switch_on,
    check_face,
    check_fold,
    excise,
    packed_deviation,
    re_excision_face,
    zone_needs_extension,
)
from pbh.horizon import HORIZON, FaceValues, HorizonReport, HorizonRow, find_horizons, near_zone
from pbh.layout import Layout
from pbh.maps import Zone
from pbh.michel import michel_flow
from pbh.monitors import MonitoredStep, StepInputs, monitor_step
from pbh.output import RunReader, RunWriter, next_snapshot_time, run_map
from pbh.readout import Epoch, first_reading, readings, starts_new_epoch
from pbh.records import StateRecord, shell_volumes
from pbh.state import State
from pbh.timestep import (
    RK4,
    Clock,
    Engine,
    Frame,
    Scheme,
    StageFluxes,
    StepAbortError,
    StepFailure,
    advance_checked,
    step_size,
)
from pbh.types import BoolArray, FloatArray

FAR_ZONE_TOLERANCE = 1e-10
"""A cell or face is in the far zone if the initial deviation from FRW there and beyond is below this fraction of the
largest initial deviation: relative, so that a perturbation given early, far below round-off of the background, still
has a far zone outside it."""

READOUT_CHECK = 0.05
"""How often, in `xi`, the read-out is tried. The reading quoted is the first qualifying one whenever it is found, so
the cadence only decides how far past it a run that stops goes."""

type Status = Literal["completed", "aborted"]
"""How a run can end: at its end time, or at an assertion with the abort recorded as a result."""

type Examination = tuple[DerivsResult, HorizonReport, FaceValues | None]
"""What examining the state at a step boundary returns: the rate as the state then stands, the finder's report, and
the face's monitors if excised."""


@dataclass(frozen=True)
class RunPaths:
    """The three files of a run, named after it."""

    config: Path
    initial: Path
    evolution: Path

    @classmethod
    def of(cls, directory: Path, name: str) -> RunPaths:
        """The paths of the run `name` in `directory`."""
        return cls(
            config=directory / f"{name}.config.yaml",
            initial=directory / f"{name}.initial.h5",
            evolution=directory / f"{name}.evolution.h5",
        )


@dataclass(frozen=True)
class RunResult:
    """How a run ended."""

    status: Status
    steps: int
    xi: float
    paths: RunPaths


class AbortError(Exception):
    """Raised inside the loop to end the run as a result; the driver records it and returns.

    `detail` joins the `abort` event's payload: for the fold monitor, the evaluation that failed (`FoldValues`).
    """

    def __init__(self, field: str, index: int, value: float, reason: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.field = field
        self.index = index
        self.value = value
        self.reason = reason
        self.detail = detail if detail is not None else {}


#: The switch-on tests whose failure means the trapped region is too narrow for the grid: under-resolution.
RESOLUTION_TESTS = ("inside_horizon", "margin_positive", "three_trapped")


def rejection_payload(failure: StepFailure) -> dict[str, Any]:
    """The `rejection` event's payload: why an attempted step was refused, where, and at which value."""
    return {"cause": failure.cause.value, "stage": failure.stage, "index": failure.index, "value": failure.value}


def clock_stalled(clock: Clock, dxi: float) -> AbortError:
    """The abort of a step that leaves the clock where it was, compensated or not: the run cannot go on in time."""
    return AbortError(
        "clock_stalled",
        -1,
        dxi,
        f"the clock stalled: a step of {dxi:.3g} no longer advances xi = {clock.xi!r} (carry {clock.carry:.3g}), "
        "even compensated",
    )


def chart_case(j: int, accepted: DerivsResult, U: float, X: float, frw2: float) -> str:
    """Why a stage or result had `Gammabar^2 <= 0` at face `j`, read from the accepted state, where it is positive.

    `Gamma^2 = 1 + U^2 - 2m/R <= 0` needs `2m/R >= 1 + U^2`. At the accepted state, in the physical normalisation (the
    tilde variables over the FRW `Gammabar`): trapped, `U + Gamma < 0`, is a trapped region the excision has not caught;
    beyond the Hubble radius, `2m/R >= 1` with `U > 0`, most likely a stage overshot its margin, since a fold of the
    chart there needs the continuation criterion of Section 4.1 to fail; and with `2m/R < 1` only a stage overshot.
    """
    d = accepted.derived
    g2 = float(d.Gammabar2[j])
    two_m_over_R = float(d.M[j]) / (X * frw2)
    trapped = U + math.sqrt(g2) < 0.0
    where = f"U = {U / math.sqrt(frw2):+.3g}, 2m/R = {two_m_over_R:.3g}, Gamma^2 = {g2 / frw2:.3g} there"
    if trapped:
        return f"a trapped region the excision has not caught ({where})"
    if two_m_over_R >= 1.0 and U > 0.0:
        return f"beyond the Hubble radius, most likely a stage overshooting its margin, a fold needing more ({where})"
    return f"a stage overshooting a thin Gammabar^2 margin: check the step cap ({where})"


def step_abort(
    failure: StepAbortError, accepted: DerivsResult, state: State, frame: Frame, refused_switch_on: list[str]
) -> AbortError:
    """The named abort of a step no allowed halving could pass (Section 7.6), read at the accepted state it left.

    `refused_switch_on` is the failed tests of the last switch-on the driver refused, if the run is unexcised and one
    was: a trapped region the switch-on would not take, which names the case before the chart does.
    """
    f = failure.failure
    if not f.cause.is_chart:
        reason = f"no step passed after {len(failure.refused) - 1} halvings: {f.cause.value} at index {f.index}"
        return AbortError(f.cause.value, f.index, f.value, reason)
    U, X = float(state.U[f.index]), float(frame.geo.X[f.index])
    reason = f"Gammabar^2 <= 0 at face {f.index} ({f.cause.value}, stage {f.stage}): "
    reason += chart_case(f.index, accepted, U, X, frame.bg.Gammabar2)
    if refused_switch_on:
        tests = ", ".join(refused_switch_on)
        if any(t in RESOLUTION_TESTS for t in refused_switch_on):
            advice = "raise N or concentrate cells at the origin"
            reason = f"horizon under-resolved (switch-on refused by {tests}): {advice}; {reason}"
        else:
            reason = f"switch-on refused by {tests}; {reason}"  # a transition that cannot fit has aborted already
    return AbortError(f.cause.value, f.index, f.value, reason)


def far_zone_radius(initial: StateRecord) -> float:
    """The smallest face radius beyond which the initial data are FRW, relatively; the edge if none.

    FRW to `FAR_ZONE_TOLERANCE` of the largest initial deviation.
    """
    N = initial.X.size - 1
    X = initial.X
    delta_rho = np.abs(initial.delta_E / shell_volumes(X))
    delta_U = np.abs(initial.delta_U[1:] / X[1:])
    tolerance = FAR_ZONE_TOLERANCE * max(float(np.max(delta_rho)), float(np.max(delta_U)))
    perturbed_cells = np.flatnonzero(delta_rho > tolerance)
    perturbed_faces = np.flatnonzero(delta_U > tolerance) + 1
    last_cell = int(perturbed_cells[-1]) + 1 if perturbed_cells.size else 0  # the face outside the last perturbed cell
    last_face = int(perturbed_faces[-1]) if perturbed_faces.size else 0
    return float(X[min(max(last_cell, last_face) + 1, N)])


@dataclass
class Run:
    """The state of a run in progress: everything the loop changes, and the operations that change it.

    The scheme, the layout and the packed deviation are replaced together by a switch-on or a re-excision; the
    time, the step count and the formation time advance; the zones grow. `writer` is the evolution file. `whole` marks
    the cells stored whole (`storage.py`), which change only with the deviation; `carry` is the clock's (`Clock`).
    """

    config: RunConfig
    writer: RunWriter[MonitoredStep]
    sch: Scheme
    dy: FloatArray
    xi: float
    xi_form: float | None
    zones: tuple[Zone, ...]
    far_zone: float
    step: int = 0
    F_N_integral: float = 0.0
    last_refusal: list[str] = field(default_factory=lambda: list[str]())
    epoch: Epoch | None = None
    core: CoreWatch = field(default_factory=CoreWatch)
    whole: BoolArray | None = None
    carry: float = 0.0
    _state: tuple[Scheme, float, FloatArray, State] | None = field(default=None, repr=False)

    @property
    def layout(self) -> Layout:
        """The current layout: which entries are unknowns, and where the excision face is."""
        return self.sch.layout

    def state(self) -> State:
        """The state the deviation stands for, at the current time.

        Formed once for each scheme, time and deviation, which a step replaces together (the deviation is a new array,
        never written in place, and the flags change only with it), and kept: the step's record and the horizon row
        both read it.
        """
        kept = self._state
        if kept is not None and kept[0] is self.sch and kept[1] == self.xi and kept[2] is self.dy:
            return kept[3]
        state = self.sch.whole_state(self.xi, self.layout.unpack(self.dy), self.whole)
        self._state = (self.sch, self.xi, self.dy, state)
        return state

    def evaluate(self) -> DerivsResult:
        """The rate at the current state."""
        return self.sch.evaluate_deviation(self.xi, self.dy, self.whole)

    def snapshot(self) -> None:
        """Write the current state as a snapshot; with `snapshots: none`, only the initial state.

        The clock's carry is dropped with it (at most half an ulp of `xi`), since a restart from it starts with none.
        """
        if self.config.output.snapshots is SnapshotChoice.NONE and self.step > 0:
            return
        dV = self.sch.frame(self.xi).geo.dV
        self.writer.snapshot(self.step, self.xi, self.layout, self.dy, self.xi_form, self.zones, self.whole, dV)
        self.carry = 0.0

    def store(self, result: DerivsResult) -> DerivsResult:
        """Move the cells across the storage lines at this step boundary (`storage.py`); the rate as they then stand.

        A move changes no state, only how it is stored, but the stage forms a cell's fields from what is stored, so
        the rate is evaluated again.
        """
        moved = storage.switched(self.dy, self.whole, result.derived.rho, self.sch.frame(self.xi).geo.dV, self.layout)
        if moved is None:
            return result
        self.dy, self.whole = moved
        return self.evaluate()

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        """Record an event at the current step and time."""
        self.writer.event(self.step, self.xi, kind, payload)

    def isolation(self, xi: float, r: float) -> dict[str, Any]:
        """The domains that keep the radius `r` out of the outer face's reach until `xi` (`causal.py`)."""
        return isolation(self.sch.eos, self.config.evolution.began, xi, r, self.config.grid.Rtilde_max)

    # --- the operations of Section 8.2 and 8.3 ---

    def remap(self, layout: Layout, state: State) -> DerivsResult:
        """Replace the scheme, the layout and the deviation for `state` on the current zones; the rate there."""
        self.sch = self.config.scheme(run_map(self.config, self.zones), layout, self.sch.engine)
        geo = self.sch.frame(self.xi).geo
        if self.whole is not None:  # the cells stored whole stay so, the excised ones drop out
            whole = self.whole.copy()
            whole[: layout.j_e] = False
            self.whole = storage.any_whole(whole)
        self.dy = packed_deviation(state, geo, layout, geo.dV, self.whole)
        return self.evaluate()

    def switch_on(self, attempt: SwitchAttempt, state: State) -> DerivsResult:
        """Throw the switch: snapshot the state unexcised with the new zone, pin the zone, drop the interior."""
        pinned_now = bool(self.zones) and self.zones[-1].xi_on == self.xi  # restarted from the switch-on snapshot
        zone = self.zones[-1] if pinned_now else attempt.zone(self.xi, self.config.excision.tau_on)
        if not pinned_now:
            self.zones = (*self.zones, zone)
            self.snapshot()  # the state the switch is thrown on: the unexcised continuation starts here
        excised, layout = excise(state, self.layout, attempt.j_e)
        self.event(
            "switch_on",
            {
                "xi_on": self.xi,
                "x_AH_on": attempt.x_AH,
                "x_e": attempt.x_e,
                "j_e": attempt.j_e,
                "x_t": zone.x_t,
                "Delta_t": zone.Delta_t,
                "M_je": excised.M_e,
                "repeated": layout.j_e > 0 and self.layout.excised,
            },
        )
        return self.remap(layout, excised)

    def re_excise(self, j_e: int, state: State, trigger: str) -> DerivsResult:
        """Move the face outward to `j_e`."""
        excised, layout = excise(state, self.layout, j_e)
        payload = {"from": self.layout.j_e, "to": j_e, "M_je_before": state.M_e, "M_je_after": excised.M_e}
        self.event("re_excision", {**payload, "trigger": trigger})
        return self.remap(layout, excised)

    def extend_zone(self, attempt: SwitchAttempt, state: State) -> DerivsResult:
        """Pin a further zone from a passed attempt, moving the face out to its face if that is further."""
        zone = attempt.zone(self.xi, self.config.excision.tau_on)
        self.zones = (*self.zones, zone)
        self.event("zone_extension", {"xi_on": self.xi, "x_AH": attempt.x_AH, "x_t": zone.x_t, "Delta_t": zone.Delta_t})
        if attempt.j_e > self.layout.j_e:
            return self.re_excise(attempt.j_e, state, trigger="zone_extension")
        return self.remap(self.layout, state)

    def refuse(self, attempt: SwitchAttempt, kind: str) -> None:
        """Record a refused attempt, once per change of the reasons, so that a long refusal is not a long log."""
        if attempt.failed != self.last_refusal:
            self.last_refusal = attempt.failed
            self.event(kind, {"failed": attempt.failed, "j_e": attempt.j_e, "x_AH": attempt.x_AH, "mu": attempt.mu})

    # --- the horizon row and the mass ---

    def record_horizon(self, report: HorizonReport, result: DerivsResult, face: FaceValues | None) -> HorizonRow:
        """Write the step's horizon row, with the near-zone monitors, and add its apparent horizon to the epoch.

        Then, once excised, assert the fold monitor: a violation ends the run with the row written, and its evaluation
        is the `abort` event's payload too (output specification, Section 7).
        """
        frame = self.sch.frame(self.xi)
        state, eos = self.state(), self.sch.eos
        near = near_zone(state, result.derived, frame.geo, report, eos, self.layout, self.xi, self.sch.engine)
        row = HorizonRow.of(self.step, self.xi, report, self.zones[-1].inner_edge if self.zones else None, near, face)
        self.writer.horizon_row(row)
        a = report.apparent
        if a is not None:
            if self.epoch is not None and starts_new_epoch(report, self.epoch.X_AH):
                previous = {"M_AH_previous": self.epoch.M_AH[-1], "X_AH_previous": self.epoch.X_AH}
                self.event("epoch", {"M_AH": report.M_AH, "X_AH": a.X, **previous})
                self.epoch = Epoch(xi_start=self.xi)
            if self.epoch is None:
                self.epoch = Epoch(xi_start=self.xi_form if self.xi_form is not None else self.xi)
            self.epoch.add(self.xi, report.M_AH, a.X)
        if face is not None:
            try:
                check_fold(face.fold)
            except ExcisionError as failure:
                detail = {"fold": dataclasses.asdict(face.fold)}
                raise AbortError(failure.check, failure.face, failure.value, str(failure), detail) from failure
        return row

    def watch_core(self, rho_0: float, row: HorizonRow) -> bool:
        """Before formation, add the step to the core's series and test it for a bounce when due; whether to stop."""
        core = self.core
        if self.xi_form is not None or core.bounced:
            return False
        core.add(self.xi, rho_0, row.core_margin)
        if self.xi < core.checked + READOUT_CHECK:
            return False
        core.checked = self.xi
        history = core.history()
        if history.bounce_xi is None:
            return False
        core.bounced = True
        self.event("bounce", dataclasses.asdict(history))
        return self.config.evolution.stop_on_bounce

    def read_mass(self, row: HorizonRow) -> bool:
        """Try the read-out on the epoch's series when due, and record it once read; whether the run should stop."""
        epoch, readout = self.epoch, self.config.readout
        if epoch is None or epoch.read or self.xi < epoch.checked + READOUT_CHECK:
            return False
        settings = readout.build()
        if self.xi < epoch.xi_start + settings.floor + 0.5 * settings.window:  # no reading can qualify yet
            return False
        epoch.checked = self.xi
        eos = self.sch.eos
        r = readings(np.array(epoch.xi), np.array(epoch.M_AH), eos, settings)
        i = first_reading(r, epoch.xi_start, settings)
        if i is None:
            return False
        epoch.read = True
        efficiency = float(r.efficiency[i])
        xi_reading, M_AH = float(r.xi[i]), float(r.M_AH[i])  # the apparent horizon then: M_AH = e^(alpha xi) X_AH / 2
        michel = michel_flow(np.array([HORIZON, eos.sonic_radius_over_mass])) if eos.is_radiation else None
        self.event(
            "readout",
            {
                "epoch_start": epoch.xi_start,
                "xi_reading": xi_reading,
                "M_est": float(r.M_est[i]),
                "bar": float(r.bar[i]),
                "systematic": float(r.systematic[i]),
                "M_AH": M_AH,
                "omega": float(r.omega[i]),
                "lambda_c_eps": float(r.lambda_c_eps[i]),
                "efficiency": efficiency,
                "efficiency_flag": abs(efficiency - 1.0) > readout.efficiency_tolerance,
                "near_zone": {
                    "lapse": [row.lapse_AH, row.lapse_sonic],
                    "v": [row.v_AH, row.v_sonic],
                    "rho": [row.rho_AH, row.rho_sonic],
                    "michel_lapse": michel.N.tolist() if michel is not None else None,
                    "michel_v": michel.v.tolist() if michel is not None else None,
                    "michel_rho": michel.compression.tolist() if michel is not None else None,
                },
                "outflow_margin": row.mu,
                "min_lapse": row.min_lapse,
                "isolation": self.isolation(xi_reading, 2.0 * M_AH * math.exp(-eos.alpha_float * xi_reading)),
                "settings": readout.model_dump(),
            },
        )
        return readout.stop

    # --- what is done with the state at a step boundary ---

    def find(self, state: State, result: DerivsResult) -> HorizonReport:
        """The horizon finder on the state, on the current layout and map."""
        frame = self.sch.frame(self.xi)
        return find_horizons(
            state,
            result.derived,
            frame.geo,
            frame.bg,
            self.sch.eos,
            self.sch.map,
            self.layout,
            self.xi,
            self.sch.engine,
        )

    def examine(self, state: State, result: DerivsResult) -> Examination:
        """Examine the state at a step boundary: the finder, formation, and the excision decisions.

        Returns the rate at the state as it stands afterwards (the same, or on the new layout), the finder's report,
        and the face's monitors if excised.
        """
        report = self.find(state, result)
        if report.outer_face_trapped:
            raise AbortError("outer_face_trapped", self.layout.N, float(report.h[-1]), "the outer face is trapped")
        if self.xi_form is None and report.trapped_faces > 0:
            self.xi_form = self.xi
            a = report.apparent
            assert a is not None  # a trapped face has an outer boundary once face N is untrapped
            payload = {"j_star": a.j, "x_AH": a.x, "X_AH": a.X, "M_AH": report.M_AH}
            self.event("formation", {**payload, "isolation": self.isolation(self.xi, a.X)})
        if self.layout.excised:  # with or without a trapped face: without one, the face assertion ends the run
            return self.excised_step(state, result, report)
        if not self.config.excision.enabled or report.trapped_faces == 0:
            return result, report, None
        return self.consider_switch_on(state, result, report)

    def consider_switch_on(self, state: State, result: DerivsResult, report: HorizonReport) -> Examination:
        """Unexcised with a trapped face: attempt the switch, and throw it if the four tests pass."""
        frame = self.sch.frame(self.xi)
        excision = self.config.excision
        pinned_now = bool(self.zones) and self.zones[-1].xi_on == self.xi
        earlier = self.zones[:-1] if pinned_now else self.zones
        attempt = attempt_switch_on(
            report,
            state,
            result.derived,
            frame.geo,
            self.sch.eos,
            self.layout,
            excision,
            earlier,
            self.sch.map,
            self.xi,
        )
        self.fail_if_no_room(attempt)
        if not attempt.passed:
            self.refuse(attempt, "switch_attempt")
            return result, report, None
        result = self.switch_on(attempt, state)
        return self.face_checked(result, report)

    def excised_step(self, state: State, result: DerivsResult, report: HorizonReport) -> Examination:
        """Excised: assert the face, then re-excise or extend the zone if either is due."""
        excision = self.config.excision
        frame = self.sch.frame(self.xi)
        face = self.assert_face(state, result, report)
        if excision.eta_r is not None:
            j_new = re_excision_face(report, self.layout, excision.eta_r)
            if j_new is not None:
                result = self.re_excise(j_new, state, trigger="accretion")
                return self.face_checked(result, report)
        if zone_needs_extension(report, self.zones, excision.zone_extension_at):
            attempt = attempt_switch_on(
                report,
                state,
                result.derived,
                frame.geo,
                self.sch.eos,
                self.layout,
                excision,
                self.zones,
                self.sch.map,
                self.xi,
            )
            self.fail_if_no_room(attempt)
            if attempt.passed:
                return self.face_checked(self.extend_zone(attempt, state), report)
            self.refuse(attempt, "zone_attempt")
        return result, report, face

    def fail_if_no_room(self, attempt: SwitchAttempt) -> None:
        """End the run if the map's transition cannot fit on the grid: the horizon only grows, so it never will."""
        if not attempt.transition_fits:
            raise AbortError(
                "transition_fits",
                attempt.j_star,
                attempt.X_out,
                f"the pinned zone's transition must reach X = {attempt.X_out:.4g}, beyond the static part of the grid, "
                f"which starts at X = {attempt.X_static:.4g}: enlarge Rtilde_max",
            )

    def face_checked(self, result: DerivsResult, report: HorizonReport) -> Examination:
        """After the layout changed: the finder again on the new layout, and the face asserted there."""
        state = self.state()
        report = self.find(state, result)
        return result, report, self.assert_face(state, result, report)

    def assert_face(self, state: State, result: DerivsResult, report: HorizonReport) -> FaceValues:
        """The two assertions of every excised step, an abort if either fails.

        The fold monitor is formed with them and asserted once the row is written (`record_horizon`).
        """
        frame = self.sch.frame(self.xi)
        try:
            return check_face(
                report,
                state,
                result.derived,
                result.speeds,
                float(result.F[self.layout.j_e]),
                frame.geo,
                frame.bg,
                self.sch.eos,
                self.layout,
                self.xi,
            )
        except ExcisionError as failure:
            raise AbortError(failure.check, failure.face, failure.value, str(failure)) from failure


def epoch_history(reader: RunReader, xi: float) -> Epoch | None:
    """The epoch a run is in at `xi`, with its series of `M_AH` before `xi`, from the run's file; `None` if unformed.

    The epoch began at the last `formation` or `epoch` event at or before `xi`; a restart from a snapshot at `xi` is
    handed it, so that it reads the mass the uninterrupted run would read.
    """
    starts = [e.xi for e in reader.events if e.kind in ("formation", "epoch") and e.xi <= xi]
    if not starts:
        return None
    epoch = Epoch(xi_start=starts[-1])
    h = reader.horizon
    for t, M, X in zip(h["xi"], h["M_AH"], h["X_AH"], strict=True):
        if epoch.xi_start <= float(t) < xi and np.isfinite(float(M)):
            epoch.add(float(t), float(M), float(X))
    return epoch


def core_watch(reader: RunReader, xi: float) -> CoreWatch:
    """The core's watch as the uninterrupted run held it after its step at `xi`, from the run's file, for a restart.

    Its series is the central density of the step record and the core margin of the horizon table, joined on the step,
    for every step up to `xi` before any formation; it is marked bounced if a `bounce` event came at or before `xi`,
    and the times it was tested are replayed with the cadence `watch_core` uses, so that the restarted run tests it at
    the same steps and records the same bounce at the same step as the uninterrupted one would.
    """
    watch = CoreWatch()
    formed = [e.xi for e in reader.events if e.kind == "formation"]
    watch.bounced = any(e.kind == "bounce" and e.xi <= xi for e in reader.events)
    margins = dict(zip(reader.horizon["step"], reader.horizon["core_margin"], strict=True))
    steps = reader.steps
    for step, t, rho_0 in zip(steps["step"], steps["xi"], steps["rho_0"], strict=True):
        if float(t) > xi or (formed and float(t) >= formed[0]):
            break
        watch.add(float(t), float(rho_0), float(margins[step]))
        if float(t) >= watch.checked + READOUT_CHECK:
            watch.checked = float(t)
    return watch


def run(
    config: RunConfig,
    initial: StateRecord,
    paths: RunPaths,
    history: Epoch | None = None,
    core: CoreWatch | None = None,
    engine: Engine = Engine.PYTHON,
    half_of: str | None = None,
) -> RunResult:
    """Run the configuration from the initial state and write the run's files; see the module docstring.

    `history` is the epoch the initial state is in, with its series of `M_AH`, and `core` the core's watch
    (`core_watch`), when it continues another run. `engine` evaluates the stages, and is recorded in both files.
    `half_of` names the run this one is the half-resolution companion of (`pair.py`), recorded in the configuration's
    provenance. A configuration without `xi_start` takes it from the initial data, and the run's files record it; it
    must then give one if the data are a snapshot of another run (their provenance names its `source`), since the
    snapshot's time is not when that run began, which the causal bookkeeping counts from.
    """
    if config.evolution.xi_start is None:  # a fresh run begins with its data; a restart passes its source's start
        if "source" in initial.provenance:  # a snapshot: its time is not when the run it continues began
            raise ValueError(
                f"the initial state is a snapshot of {initial.provenance['source']} and the configuration gives no "
                "evolution.xi_start: give the start of the run it continues, or use `pbh restart`, which passes it"
            )
        config = config.starting_at(initial.xi)
    if initial.j_e > 0 and not config.excision.enabled:
        raise ValueError(
            f"the initial state is excised (j_e = {initial.j_e}) but the configuration disables excision: an excised "
            "state can only continue with excision on (excision.enabled: true)"
        )
    layout = Layout(config.grid.N, j_e=initial.j_e)
    sch = config.scheme(run_map(config, initial.zones), layout, engine)
    xi = initial.xi
    if not np.allclose(initial.X, sch.frame(xi).geo.X[: layout.N + 1], rtol=1e-12, atol=0.0):
        raise ValueError("the initial data are not sampled on the grid the configuration builds at their time")
    output = config.output
    xi_end = config.evolution.xi_end
    if not xi_end > xi:
        raise ValueError(f"the initial time {xi} is not before the end {xi_end}")
    if xi < config.evolution.began:
        raise ValueError(
            f"the initial time {xi} is before the configuration's xi_start = {config.evolution.began}: a run starts "
            "at xi_start, or continues one that did"
        )
    save(config, paths.config, engine, half_of)
    cap = config.stepping.cap(sch.eos)
    weights = tuple(float(b) for b in RK4.b)
    scale = float(np.max(np.abs(sch.frw(xi))))  # the state's scale, for the companion estimate

    with RunWriter(paths.evolution, config, layout.N, row_type=MonitoredStep, engine=engine) as out:
        r = Run(
            config,
            out,
            sch,
            layout.pack(initial.stored),
            xi,
            initial.xi_form,
            initial.zones,
            far_zone_radius(initial),
            epoch=history,
            core=core if core is not None else CoreWatch(),
            whole=initial.whole,
        )
        try:
            if not bool(np.all(np.isfinite(r.dy))):  # a failure at the accepted state is an abort, never a retry
                raise AbortError("state", -1, float("nan"), "the initial state is not finite")
            result, report, face = r.examine(r.state(), r.store(r.evaluate()))
            r.snapshot()
            read = r.read_mass(r.record_horizon(report, result, face))
            scheduled = output.snapshots is SnapshotChoice.ALL
            rho_0 = float(result.derived.rho[r.layout.j_e])
            next_snapshot = next_snapshot_time(r.xi, r.xi_form, output, rho_0) if scheduled else math.inf
            at_snapshot = True
            bounced = False
            while r.xi < xi_end and not read and not bounced:
                # 1. the step: Courant or cap, clipped to land exactly on the end and, unless the configuration lets
                # the stepper run free, on the next snapshot time
                choice = step_size(result, r.sch.frame(r.xi).geo, r.layout, config.stepping.courant_number, cap)
                dxi, limit = choice.dxi, choice.limit.value
                landing = min(next_snapshot, xi_end) if output.clip_to_snapshots else xi_end
                if r.xi + dxi >= landing - 1e-12 * max(1.0, abs(landing)):
                    dxi, limit = landing - r.xi, "output_clip"
                # 2. the checked step: every stage and the result inside the domain, or a halved step (Section 7.6)
                layout = r.layout
                before = StageFluxes.of(result, layout)
                clipped = limit == "output_clip"
                land = landing if clipped else None
                accepted = advance_checked(r.sch, r.xi, r.dy, dxi, result, land, r.carry, r.whole)
                for failure in accepted.refused:
                    r.event("rejection", rejection_payload(failure))
                if accepted.refused:
                    dxi, limit = accepted.dxi, "halved"
                clock, arrived = Clock(r.xi, r.carry), accepted.clock
                if arrived == clock:
                    raise clock_stalled(clock, dxi)
                dy_new, stages = accepted.dy, accepted.stages
                result = accepted.result
                # 3. the record of the step, after the cells are moved across the storage lines
                r.step += 1
                change = layout.pack(result.stored_rate) - stages[-1].k
                rate_change = float(np.max(np.abs(change)))
                r.xi, r.carry, r.dy = arrived.xi, arrived.carry, dy_new
                result = r.store(result)
                state_new = r.state()
                at_snapshot = r.xi >= next_snapshot  # exactly on it when clipped, the first step past it when not
                frame = r.sch.frame(r.xi)
                row = monitor_step(
                    StepInputs(
                        step=r.step,
                        xi=r.xi,
                        dxi=dxi,
                        limit=limit,
                        halvings=len(accepted.refused),
                        state=state_new,
                        geo=frame.geo,
                        bg=frame.bg,
                        result=result,
                        stages=[s.fluxes for s in stages],
                        weights=weights,
                        delta_M_total_before=before.delta_M_total,
                        F_N_integral_before=r.F_N_integral,
                        rate_change=rate_change / scale,
                        far_zone_from=r.far_zone,
                        engine=r.sch.engine,
                    ),
                    r.sch.eos,
                    layout,
                    full=at_snapshot or output.monitor_every_step,
                )
                r.F_N_integral = row.F_N_integral
                out.step(row)
                # 4. examine the state arrived at; the horizon row, the snapshot, the flush
                formed_before = r.xi_form is not None
                result, report, face = r.examine(state_new, result)
                horizon_row = r.record_horizon(report, result, face)
                read = r.read_mass(horizon_row)
                bounced = r.watch_core(row.rho_0, horizon_row)
                formed_now = r.xi_form is not None and not formed_before
                if at_snapshot or (formed_now and not scheduled):  # the formation is a milestone
                    r.snapshot()
                    at_snapshot = True
                if scheduled and (at_snapshot or formed_now):
                    next_snapshot = next_snapshot_time(r.xi, r.xi_form, output, row.rho_0)  # re-planned at formation
                if r.step % output.flush_every == 0:
                    out.flush()
            if not at_snapshot:
                r.snapshot()  # the end is always a snapshot, to continue from
            reason = {"reason": "the mass was read"} if read else {"reason": "the core bounced"} if bounced else {}
            out.close(r.step, r.xi, "completed", **reason, isolation=r.isolation(r.xi, 0.0))
            return RunResult(status="completed", steps=r.step, xi=r.xi, paths=paths)
        except NotHyperbolicError as failure:
            abort = AbortError(failure.field, failure.index, failure.value, str(failure))
        except StepAbortError as failure:
            for refused in failure.refused:
                r.event("rejection", rejection_payload(refused))
            refused_switch_on = r.last_refusal if not r.layout.excised else []
            abort = step_abort(failure, r.evaluate(), r.state(), r.sch.frame(r.xi), refused_switch_on)
        except AbortError as failure:
            abort = failure
        r.event("abort", {"field": abort.field, "index": abort.index, "value": abort.value, **abort.detail})
        r.snapshot()  # the last good state
        out.close(r.step, r.xi, "aborted", reason=abort.reason, isolation=r.isolation(r.xi, 0.0))
        return RunResult(status="aborted", steps=r.step, xi=r.xi, paths=paths)

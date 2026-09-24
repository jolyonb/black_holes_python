"""The driver: one loop that runs a configuration from an initial state to its end, writing the run's files.

A run is `run(config, initial, paths)`. It builds the `Scheme` from the configuration and the initial record, which
carries the map's zones and the formation time if it comes from a snapshot after formation, checks that the record
sits on the grid that gives, saves the configuration with its provenance, and then steps:

1. evaluate the rate at the state (the first stage of the step) and choose the step: the smaller of the Courant
   step and the cap (eq:num:cfl), clipped to land exactly on the next snapshot time and on the end;
2. take the Runge-Kutta step in deviation form, keeping every stage's result for the monitors;
3. check that the state is finite and record the step's monitors, the full row when the step ends on a snapshot or
   the configuration asks for it every step;
4. examine the state arrived at: run the horizon finder, record formation at the first trapped face, and, with
   excision enabled, throw the switch when its four tests pass, assert the face every excised step, move the face
   outward by re-excision, and pin a further zone when the horizon approaches the transition; record the horizon
   row, write the snapshot when due, flush the step record on its cadence;
5. read the mass (Section 8.5, `readout.py`): the apparent-horizon mass of every step is collected by epoch, a new
   one beginning when a trapped region appears outside the apparent horizon (an `epoch` event), and every
   `READOUT_CHECK` in `xi` from the floor on the read-out is tried on the epoch's series; the first reading with its
   bar below the target is recorded as a `readout` event, with the near-zone monitors beside the Michel values, the
   outflow margin, the minimum lapse and a flag if the efficiency is not yet near Michel's, and the run ends there
   if the configuration says so.

The initial state is examined the same way before the first step, so that a run restarted from a snapshot makes
the decisions the uninterrupted run made at that state; a restart hands the run its epoch's history of `M_AH` from
the source file (`epoch_history`), so that it reads the mass the uninterrupted run would read. The switch-on writes
a snapshot of the state it is thrown on, unexcised and already carrying the new zone: the same collapse continued
from it with excision off runs on the same grid, which is the comparison Section 8.3 asks for.

An abort is a result, not an exception: if a stage finds the state outside the hyperbolic domain, the step
produces a non-finite state, the outer face is trapped, or an excision assertion fails, the driver records an
`abort` event naming what failed, writes a snapshot of the last good state, and ends the run with the status
`aborted`. The far-zone radius of the monitors is read off the initial data.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pbh.config import RunConfig, save
from pbh.derived import NotHyperbolicError
from pbh.equations import DerivsResult
from pbh.excision import (
    ExcisionError,
    SwitchAttempt,
    attempt_switch_on,
    check_face,
    excise,
    packed_deviation,
    re_excision_face,
    zone_needs_extension,
)
from pbh.horizon import HORIZON, FaceValues, HorizonReport, HorizonRow, find_horizons, near_zone
from pbh.layout import Layout
from pbh.maps import Zone
from pbh.michel import michel_flow
from pbh.monitors import MonitoredStep, StageFluxes, StepInputs, monitor_step
from pbh.output import RunReader, RunWriter, next_snapshot_time, run_map
from pbh.readout import Epoch, first_reading, readings, starts_new_epoch
from pbh.records import StateRecord, shell_volumes
from pbh.state import State, is_finite
from pbh.timestep import Scheme, advance_with_stages, step_size
from pbh.types import FloatArray

FAR_ZONE_TOLERANCE = 1e-10
"""A cell or face is in the far zone if the initial deviation from FRW there and beyond is below this."""

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
    """Raised inside the loop to end the run as a result; the driver records it and returns."""

    def __init__(self, field: str, index: int, value: float, reason: str) -> None:
        super().__init__(reason)
        self.field = field
        self.index = index
        self.value = value
        self.reason = reason


def far_zone_radius(initial: StateRecord) -> float:
    """The smallest face radius beyond which the initial data are FRW to `FAR_ZONE_TOLERANCE`; the edge if none."""
    N = initial.X.size - 1
    X = initial.X
    delta_rho = np.abs(initial.delta_E / shell_volumes(X))
    delta_U = np.abs(initial.delta_U[1:] / X[1:])
    perturbed_cells = np.flatnonzero(delta_rho > FAR_ZONE_TOLERANCE)
    perturbed_faces = np.flatnonzero(delta_U > FAR_ZONE_TOLERANCE) + 1
    last_cell = int(perturbed_cells[-1]) + 1 if perturbed_cells.size else 0  # the face outside the last perturbed cell
    last_face = int(perturbed_faces[-1]) if perturbed_faces.size else 0
    return float(X[min(max(last_cell, last_face) + 1, N)])


@dataclass
class Run:
    """The state of a run in progress: everything the loop changes, and the operations that change it.

    The scheme, the layout and the packed deviation are replaced together by a switch-on or a re-excision; the
    time, the step count and the formation time advance; the zones grow. `writer` is the evolution file.
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

    @property
    def layout(self) -> Layout:
        """The current layout: which entries are unknowns, and where the excision face is."""
        return self.sch.layout

    def state(self) -> State:
        """The state the deviation stands for, at the current time."""
        return self.layout.unpack(self.sch.frw(self.xi) + self.dy)

    def evaluate(self) -> DerivsResult:
        """The rate at the current state."""
        return self.sch.evaluate_deviation(self.xi, self.dy)

    def snapshot(self) -> None:
        """Write the current state as a snapshot."""
        self.writer.snapshot(self.step, self.xi, self.layout, self.dy, self.xi_form, self.zones)

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        """Record an event at the current step and time."""
        self.writer.event(self.step, self.xi, kind, payload)

    # --- the operations of Section 8.2 and 8.3 ---

    def remap(self, layout: Layout, state: State) -> DerivsResult:
        """Replace the scheme, the layout and the deviation for `state` on the current zones; the rate there."""
        self.sch = self.config.scheme(run_map(self.config, self.zones), layout)
        geo = self.sch.frame(self.xi).geo
        self.dy = packed_deviation(state, geo, layout, geo.dV)
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
        """Write the step's horizon row, with the near-zone monitors, and add its apparent horizon to the epoch."""
        frame = self.sch.frame(self.xi)
        near = near_zone(self.state(), result.derived, frame.geo, report, self.sch.eos, self.layout, self.xi)
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
        return row

    def read_mass(self, row: HorizonRow) -> bool:
        """Try the read-out on the epoch's series when due, and record it once read; whether the run should stop."""
        epoch, readout = self.epoch, self.config.readout
        settings = readout.build()
        if epoch is None or epoch.read or self.xi < epoch.checked + READOUT_CHECK:
            return False
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
        michel = michel_flow(np.array([HORIZON, eos.sonic_radius_over_mass])) if eos.is_radiation else None
        self.event(
            "readout",
            {
                "epoch_start": epoch.xi_start,
                "xi_reading": float(r.xi[i]),
                "M_est": float(r.M_est[i]),
                "bar": float(r.bar[i]),
                "systematic": float(r.systematic[i]),
                "M_AH": float(r.M_AH[i]),
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
                "settings": readout.model_dump(),
            },
        )
        return readout.stop

    # --- what is done with the state at a step boundary ---

    def find(self, state: State, result: DerivsResult) -> HorizonReport:
        """The horizon finder on the state, on the current layout and map."""
        frame = self.sch.frame(self.xi)
        return find_horizons(
            state, result.derived, frame.geo, frame.bg, self.sch.eos, self.sch.map, self.layout, self.xi
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
            self.event("formation", {"j_star": a.j, "x_AH": a.x, "X_AH": a.X, "M_AH": report.M_AH})
        if not self.config.excision.enabled or report.trapped_faces == 0:
            return result, report, None
        if not self.layout.excised:
            return self.consider_switch_on(state, result, report)
        return self.excised_step(state, result, report)

    def consider_switch_on(self, state: State, result: DerivsResult, report: HorizonReport) -> Examination:
        """Unexcised with a trapped face: attempt the switch, and throw it if the four tests pass."""
        frame = self.sch.frame(self.xi)
        excision = self.config.excision
        pinned_now = bool(self.zones) and self.zones[-1].xi_on == self.xi
        earlier = self.zones[:-1] if pinned_now else self.zones
        attempt = attempt_switch_on(
            report, state, result.derived, frame.geo, self.sch.eos, self.layout, excision, earlier
        )
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
                report, state, result.derived, frame.geo, self.sch.eos, self.layout, excision, self.zones
            )
            if attempt.passed:
                return self.face_checked(self.extend_zone(attempt, state), report)
            self.refuse(attempt, "zone_attempt")
        return result, report, face

    def face_checked(self, result: DerivsResult, report: HorizonReport) -> Examination:
        """After the layout changed: the finder again on the new layout, and the face asserted there."""
        state = self.state()
        report = self.find(state, result)
        return result, report, self.assert_face(state, result, report)

    def assert_face(self, state: State, result: DerivsResult, report: HorizonReport) -> FaceValues:
        """The two assertions of every excised step, an abort if either fails."""
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


def run(config: RunConfig, initial: StateRecord, paths: RunPaths, history: Epoch | None = None) -> RunResult:
    """Run the configuration from the initial state and write the run's files; see the module docstring.

    `history` is the epoch the initial state is in, with its series of `M_AH`, when it continues another run.
    """
    layout = Layout(config.grid.N, j_e=initial.j_e)
    sch = config.scheme(run_map(config, initial.zones), layout)
    xi = initial.xi
    if not np.allclose(initial.X, sch.frame(xi).geo.X[: layout.N + 1], rtol=1e-12, atol=0.0):
        raise ValueError("the initial data are not sampled on the grid the configuration builds at their time")
    output = config.output
    xi_end = config.evolution.xi_end
    if not xi_end > xi:
        raise ValueError(f"the initial time {xi} is not before the end {xi_end}")
    save(config, paths.config)
    cap = config.stepping.cap(sch.eos)
    weights = tuple(float(b) for b in config.stepping.integrator.tableau.b)
    scale = float(np.max(np.abs(sch.frw(xi))))  # the state's scale, for the companion estimate

    with RunWriter(paths.evolution, config, layout.N, row_type=MonitoredStep) as out:
        r = Run(
            config,
            out,
            sch,
            layout.pack(initial.deviation),
            xi,
            initial.xi_form,
            initial.zones,
            far_zone_radius(initial),
            epoch=history,
        )
        try:
            result, report, face = r.examine(r.state(), r.evaluate())
            r.snapshot()
            read = r.read_mass(r.record_horizon(report, result, face))
            next_snapshot = next_snapshot_time(r.xi, r.xi_form, output)
            at_snapshot = True
            while r.xi < xi_end and not read:
                # 1. the step: Courant or cap, clipped to land exactly on the next snapshot time or the end
                choice = step_size(result, r.sch.frame(r.xi).geo, r.layout, config.stepping.courant_number, cap)
                dxi, limit = choice.dxi, choice.limit.value
                landing = min(next_snapshot, xi_end)
                if r.xi + dxi >= landing - 1e-12 * max(1.0, abs(landing)):
                    dxi, limit = landing - r.xi, "output_clip"
                # 2. the stages, and the state arrived at
                layout = r.layout
                before = StageFluxes.of(result, layout)
                dy_new, stages = advance_with_stages(r.sch, config.stepping.integrator, r.xi, r.dy, dxi, first=result)
                xi_new = landing if dxi == landing - r.xi else r.xi + dxi
                state_new = layout.unpack(r.sch.frw(xi_new) + dy_new)
                if not is_finite(state_new, layout.j_e):
                    raise AbortError("state", -1, float("nan"), "the state is not finite")
                result = r.sch.evaluate_deviation(xi_new, dy_new)
                # 3. the record of the step
                r.step += 1
                change = layout.pack(result.deviation_rate) - layout.pack(stages[-1].result.deviation_rate)
                rate_change = float(np.max(np.abs(change)))
                r.xi, r.dy = xi_new, dy_new
                at_snapshot = r.xi == next_snapshot
                frame = r.sch.frame(r.xi)
                row = monitor_step(
                    StepInputs(
                        step=r.step,
                        xi=r.xi,
                        dxi=dxi,
                        limit=limit,
                        state=state_new,
                        geo=frame.geo,
                        bg=frame.bg,
                        result=result,
                        stages=[StageFluxes.of(s.result, layout) for s in stages],
                        weights=weights,
                        delta_M_total_before=before.delta_M_total,
                        F_N_integral_before=r.F_N_integral,
                        rate_change=rate_change / scale,
                        far_zone_from=r.far_zone,
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
                read = r.read_mass(r.record_horizon(report, result, face))
                if at_snapshot:
                    r.snapshot()
                if at_snapshot or (r.xi_form is not None and not formed_before):
                    next_snapshot = next_snapshot_time(r.xi, r.xi_form, output)  # re-planned at formation
                if r.step % output.flush_every == 0:
                    out.flush()
            if not at_snapshot:
                r.snapshot()  # the end is always a snapshot, to continue from
            out.close(r.step, r.xi, "completed", **({"reason": "the mass was read"} if read else {}))
            return RunResult(status="completed", steps=r.step, xi=r.xi, paths=paths)
        except NotHyperbolicError as failure:
            abort = AbortError(failure.field, failure.index, failure.value, str(failure))
        except AbortError as failure:
            abort = failure
        r.event("abort", {"field": abort.field, "index": abort.index, "value": abort.value})
        r.snapshot()  # the last good state
        out.close(r.step, r.xi, "aborted", reason=abort.reason)
        return RunResult(status="aborted", steps=r.step, xi=r.xi, paths=paths)

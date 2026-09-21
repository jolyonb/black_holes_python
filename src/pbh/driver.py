"""The driver: one loop that runs a configuration from an initial state to its end, writing the run's files.

A run is `run(config, initial, name, directory)`. It builds the `Scheme` from the configuration, checks that the
initial state sits on the grid the configuration builds, saves the configuration with its provenance, and then
steps:

1. evaluate the rate at the state (the first stage of the step), and choose the step: the smaller of the Courant
   step and the cap (eq:num:cfl), clipped to land exactly on the next snapshot time and on the end;
2. take the Runge-Kutta step in deviation form, keeping every stage's result for the monitors;
3. check that the state is finite; record the step's monitors, the full row when the step ends on a snapshot or
   the configuration asks for it every step; write the snapshot when due; flush the step record on its cadence.

The initial state and the final state are always snapshots, whatever the schedule, so that any run can be
continued from where it stopped.

An abort is a result, not an exception: if a stage finds the state outside the hyperbolic domain, or the step
produces a non-finite state, the driver records an `abort` event naming what failed, writes a snapshot of the last
good state, and ends the run with the status `aborted`. The horizon finder and excision join this loop in Sections
8.2 and 8.3; until then the run has no formation event and the snapshot schedule stays uniform in `xi`.

The far-zone radius of the monitors is read off the initial data: the smallest radius beyond which they are FRW.
The map is the configuration's base map, blended with the zones the initial record carries if it comes from a
snapshot after a switch-on, and the formation time it carries drives the snapshot schedule; until the finder and
excision exist the driver adds neither.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from pbh.config import RunConfig, save
from pbh.derived import NotHyperbolicError
from pbh.monitors import MonitoredStep, StageFluxes, StepInputs, monitor_step
from pbh.output import RunWriter, next_snapshot_time, run_map
from pbh.records import StateRecord, shell_volumes
from pbh.state import is_finite
from pbh.timestep import advance_with_stages, step_size

FAR_ZONE_TOLERANCE = 1e-10
"""A cell or face is in the far zone if the initial deviation from FRW there and beyond is below this."""


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


type Status = Literal["completed", "aborted"]
"""How a run can end: at its end time, or at an assertion with the abort recorded as a result."""


@dataclass(frozen=True)
class RunResult:
    """How a run ended."""

    status: Status
    steps: int
    xi: float
    paths: RunPaths


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


def run(config: RunConfig, initial: StateRecord, paths: RunPaths) -> RunResult:
    """Run the configuration from the initial state and write the run's files; see the module docstring."""
    if initial.j_e != 0:
        raise ValueError("starting from an excised state arrives with excision (Section 8.3)")
    sch = config.scheme(run_map(config, initial.zones))
    layout = sch.layout
    xi_form, zones = initial.xi_form, initial.zones
    xi = initial.xi
    frame = sch.frame(xi)
    if not np.allclose(initial.X, frame.geo.X[: layout.N + 1], rtol=1e-12, atol=0.0):
        raise ValueError("the initial data are not sampled on the grid the configuration builds at their time")
    output = config.output
    xi_end = config.evolution.xi_end
    if not xi_end > xi:
        raise ValueError(f"the initial time {xi} is not before the end {xi_end}")
    save(config, paths.config)

    cap = config.stepping.cap(sch.eos)
    weights = tuple(float(b) for b in config.stepping.integrator.tableau.b)
    far_zone = far_zone_radius(initial)
    dy = layout.pack(initial.deviation)  # the integrator's variables, as the record stores them
    step = 0
    at_snapshot = True  # the initial state is the first snapshot
    next_snapshot = xi  # the initial state is the first snapshot
    F_N_integral = 0.0
    scale = float(np.max(np.abs(sch.frw(xi))))  # the state's scale, for the companion estimate

    with RunWriter(paths.evolution, config, layout.N, row_type=MonitoredStep) as out:
        result = sch.evaluate(xi, sch.frw(xi) + dy)
        out.snapshot(step, xi, layout, dy, xi_form, zones)
        next_snapshot = next_snapshot_time(xi, xi_form, output)
        while xi < xi_end:
            # 1. the step: Courant or cap, clipped to land exactly on the next snapshot time or the end
            choice = step_size(result, sch.frame(xi).geo, layout, config.stepping.courant_number, cap)
            dxi, limit = choice.dxi, choice.limit.value
            landing = min(next_snapshot, xi_end)
            if xi + dxi >= landing - 1e-12 * max(1.0, abs(landing)):
                dxi, limit = landing - xi, "output_clip"
            # 2. the stages, and the state arrived at
            M_total_before = StageFluxes.of(result, layout.unpack(sch.frw(xi) + dy), layout).M_total
            try:
                dy_new, stages = advance_with_stages(sch, config.stepping.integrator, xi, dy, dxi, first=result)
                xi_new = landing if dxi == landing - xi else xi + dxi
                state_new = layout.unpack(sch.frw(xi_new) + dy_new)
                if not is_finite(state_new, layout.j_e):
                    raise NotHyperbolicError("state", -1, float("nan"))
                result_new = sch.evaluate(xi_new, sch.frw(xi_new) + dy_new)
            except NotHyperbolicError as failure:
                out.event(step, xi, "abort", {"field": failure.field, "index": failure.index, "value": failure.value})
                out.snapshot(step, xi, layout, dy, xi_form, zones)  # the last good state
                out.close(step, xi, "aborted", reason=str(failure))
                return RunResult(status="aborted", steps=step, xi=xi, paths=paths)
            # 3. the record
            step += 1
            rate_change = float(np.max(np.abs(layout.pack(result_new.rate) - layout.pack(stages[-1].result.rate))))
            xi, dy, result = xi_new, dy_new, result_new
            at_snapshot = xi == next_snapshot
            frame = sch.frame(xi)
            row = monitor_step(
                StepInputs(
                    step=step,
                    xi=xi,
                    dxi=dxi,
                    limit=limit,
                    state=state_new,
                    geo=frame.geo,
                    bg=frame.bg,
                    result=result,
                    stages=[StageFluxes.of(s.result, layout.unpack(s.y), layout) for s in stages],
                    weights=weights,
                    M_total_before=M_total_before,
                    F_N_integral_before=F_N_integral,
                    rate_change=rate_change / scale,
                    far_zone_from=far_zone,
                ),
                sch.eos,
                layout,
                sch.settings,
                full=at_snapshot or output.monitor_every_step,
            )
            F_N_integral = row.F_N_integral
            out.step(row)
            if at_snapshot:
                out.snapshot(step, xi, layout, dy, xi_form, zones)
                next_snapshot = next_snapshot_time(xi, xi_form, output)
            if step % output.flush_every == 0:
                out.flush()
        if not at_snapshot:
            out.snapshot(step, xi, layout, dy, xi_form, zones)  # the end is always a snapshot, to continue from
        out.close(step, xi, "completed")
    return RunResult(status="completed", steps=step, xi=xi, paths=paths)

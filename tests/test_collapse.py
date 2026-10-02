"""Tests of pbh.collapse: the peak of the physical central density, and the bounce as positive evidence."""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pbh.cli import main
from pbh.collapse import (
    BOUNCE_FALL,
    BOUNCE_HOLD,
    CoreWatch,
    bounce_time,
    collapse_history,
    peak_index,
    physical_density,
)
from pbh.driver import RunPaths
from pbh.output import RunReader
from pbh.summary import CoreSummary, RunSummary, describe, summarise
from pbh.types import FloatArray

XI = np.linspace(0.0, 8.0, 801)  # a step of 0.01


def central(peak_xi: float, width: float, height: float) -> FloatArray:
    """A tilde central density whose physical value falls with the background, then spikes at `peak_xi`."""
    return 1.2 + height * np.exp(2.0 * XI) * np.exp(-(((XI - peak_xi) / width) ** 2))


def test_the_physical_density_is_the_tilde_density_against_the_background_at_xi_zero():
    assert physical_density(np.array([0.0, 1.0]), np.array([2.0, 2.0])) == pytest.approx([2.0, 2.0 * math.exp(-2.0)])


def test_the_peak_is_the_largest_value_after_the_physical_density_first_rises():
    rho_phys = physical_density(XI, central(5.0, 0.2, 3.0))
    assert rho_phys[0] > rho_phys[400]  # the background's fall: largest at the start ...
    k = peak_index(rho_phys)
    assert k is not None
    assert XI[k] == pytest.approx(5.0, abs=0.02)  # ... but the collapse's peak is the spike
    assert peak_index(physical_density(XI, np.full(XI.size, 1.2))) is None  # FRW never rises


def test_the_peak_is_the_first_one_not_a_later_spike_onto_an_emptied_centre():
    # below threshold at fine origin resolution the centre empties after the bounce and later infall spikes above the
    # collapse's own peak; the peak is the one before the density first fell below half its running maximum
    rho_0 = central(4.0, 0.2, 3.0) + 8.0 * np.exp(2.0 * XI) * np.exp(-(((XI - 6.0) / 0.05) ** 2))
    rho_phys = physical_density(XI, rho_0)
    assert np.argmax(rho_phys) == pytest.approx(600, abs=2)  # the spike is the largest value
    k = peak_index(rho_phys)
    assert k is not None
    assert XI[k] == pytest.approx(4.0, abs=0.02)


def test_a_bounce_is_established_at_the_end_of_the_hold_once_the_margin_has_risen():
    rho_phys = physical_density(XI, central(5.0, 0.2, 3.0))
    margin = 1.0 - 0.8 * np.exp(-(((XI - 5.0) / 0.3) ** 2))  # smallest at the peak, recovering after
    k = peak_index(rho_phys)
    assert k is not None
    fallen = int(np.flatnonzero((rho_phys <= BOUNCE_FALL * rho_phys[k]) & (XI > XI[k]))[0])
    expected = XI[XI <= XI[fallen] + BOUNCE_HOLD][-1]
    assert bounce_time(XI, rho_phys, margin, k) == expected
    # before the hold is over the bounce is not established, and neither is it when the margin never rises
    cut = XI <= XI[fallen] + 0.3
    assert bounce_time(XI[cut], rho_phys[cut], margin[cut], k) is None
    assert bounce_time(XI, rho_phys, np.minimum.accumulate(margin), k) is None


def test_a_pause_below_half_is_not_a_bounce_but_the_fall_after_it_is():
    rho_0 = central(5.0, 0.2, 3.0) + 2.5 * np.exp(2.0 * XI) * np.exp(-(((XI - 5.6) / 0.1) ** 2))  # a second spike
    rho_phys = physical_density(XI, rho_0)
    margin = 1.0 - 0.8 * np.exp(-(((XI - 5.0) / 0.3) ** 2))
    k = peak_index(rho_phys)
    assert k is not None
    t = bounce_time(XI, rho_phys, margin, k)
    assert t is not None
    assert t > 5.6 + BOUNCE_HOLD  # the hold that the second spike interrupted does not count


def test_a_margin_still_falling_at_the_end_of_the_first_hold_is_waited_for():
    # The density falls below half its peak at about 5.3 and stays there; the margin keeps falling until 6.2 and rises
    # only after it. The first hold ends with the margin still falling, so the bounce is established later in the same
    # fall, once a hold ends with the margin above its minimum; testing only the first hold never established it.
    rho_phys = physical_density(XI, central(5.0, 0.2, 3.0))
    margin = (
        1.0 - 0.8 * np.exp(-(((XI - 6.2) / 0.3) ** 2)) * (XI < 6.2) - 0.8 * (XI >= 6.2) * np.exp(-((XI - 6.2) / 0.1))
    )
    k = peak_index(rho_phys)
    assert k is not None
    fallen = int(np.flatnonzero((rho_phys <= BOUNCE_FALL * rho_phys[k]) & (XI > XI[k]))[0])
    t = bounce_time(XI, rho_phys, margin, k)
    assert t is not None
    assert t > XI[fallen] + BOUNCE_HOLD + 0.5  # not the first hold's end: the margin was still falling there
    assert t >= 6.2  # once the margin has risen above its minimum


def test_the_history_and_the_watch_agree_on_the_same_series():
    rho_0 = central(5.0, 0.2, 3.0)
    margin = 1.0 - 0.8 * np.exp(-(((XI - 5.0) / 0.3) ** 2))
    h = collapse_history(XI, rho_0, margin)
    assert h.peak_xi == pytest.approx(5.0, abs=0.02)
    assert h.peak_rho_tilde == rho_0[int(np.argmin(np.abs(XI - h.peak_xi)))]
    assert h.margin_min == pytest.approx(0.2, abs=1e-3)
    assert h.bounce_xi is not None
    watch = CoreWatch()
    for x, r, m in zip(XI, rho_0, margin, strict=True):
        watch.add(float(x), float(r), float(m))
    assert watch.history() == h
    frw = collapse_history(XI, np.full(XI.size, 1.0), np.full(XI.size, 1.0))
    assert math.isnan(frw.peak_xi)
    assert frw.bounce_xi is None


def test_a_sub_critical_collapse_bounces_and_the_run_stops_there(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    # N = 100 keeps this in the fast suite (under a second); the core is resolved by 7 cells at the peak
    config_path = tmp_path / "sub.yaml"
    config_path.write_text(
        "grid: {N: 100, Rtilde_max: 12.0, scale: 3.0}\nevolution: {xi_end: 9.0, stop_on_bounce: true}\n"
    )
    A = 0.49 * math.e / 8.0  # peak compaction 0.49 at ell = 2: just below threshold
    initial = ["initial", "gaussian", "sub", "--config", str(config_path), "--A", f"{A:.12g}", "--ell", "2.0"]
    assert main([*initial, "--dir", str(tmp_path)]) == 0
    assert main(["run", str(config_path), "sub", "--dir", str(tmp_path)]) == 0
    reader = RunReader(RunPaths.of(tmp_path, "sub").evolution)
    assert [e.kind for e in reader.events] == ["bounce", "end"]  # no formation, no rejection
    end = reader.end
    assert end is not None
    assert end.payload["reason"] == "the core bounced"
    bounce = reader.events[0].payload
    core = summarise(reader).core
    assert core is not None
    assert core.outcome == "bounced"
    assert dataclasses.asdict(core.history) == bounce  # the summary recomputes what the run recorded
    assert 4.0 < bounce["peak_xi"] < 5.0
    assert bounce["peak_rho_tilde"] > 50.0  # a hard collapse ...
    assert bounce["margin_min"] > 0.9  # ... whose margin stays far from trapping, as near the critical solution
    assert bounce["bounce_xi"] > bounce["peak_xi"] + BOUNCE_HOLD
    assert end.xi < bounce["bounce_xi"] + 0.1  # stopped within one check of establishing it
    assert core.resolution is not None
    assert core.resolution.core_cells >= 5
    assert abs(core.resolution.xi - bounce["peak_xi"]) < 0.02  # the snapshot row nearest the peak
    # the summary, printed and exported
    capsys.readouterr()
    export = tmp_path / "sub.json"
    assert main(["summary", "sub", "--dir", str(tmp_path), "--export", str(export)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("core: bounced (established at xi = ")
    assert "cells to half the central density" in printed
    assert printed.rstrip().endswith("no horizon formed")
    data = json.loads(export.read_text())
    assert data["core"]["outcome"] == "bounced"
    assert data["core"]["history"] == bounce
    assert data["core"]["resolution"]["core_cells"] == core.resolution.core_cells
    assert data["epochs"] == []


def test_a_core_that_never_rose_is_described_as_such():
    history = collapse_history(XI, np.full(XI.size, 1.0), np.full(XI.size, 1.0))
    text = describe(RunSummary(CoreSummary("undecided", history, None), []))
    assert text.splitlines() == [
        "core: undecided",
        "  central density never rose against the background",
        "  smallest core margin 1.0000 at xi = 0.0000",
        "no horizon formed",
    ]


def test_a_run_restarted_before_its_bounce_records_the_same_bounce_at_the_same_step(tmp_path: Path):
    # The restart hands the core's watch its series from the file (driver.core_watch), and replays when it was tested,
    # so the restarted run tests at the same steps and establishes the bounce where the uninterrupted run did.
    config_path = tmp_path / "sub.yaml"
    config_path.write_text(
        "grid: {N: 100, Rtilde_max: 12.0, scale: 3.0}\nevolution: {xi_end: 9.0, stop_on_bounce: true}\n"
    )
    A = 0.49 * math.e / 8.0
    initial = ["initial", "gaussian", "sub", "--config", str(config_path), "--A", f"{A:.12g}", "--ell", "2.0"]
    assert main([*initial, "--dir", str(tmp_path)]) == 0
    assert main(["run", str(config_path), "sub", "--dir", str(tmp_path)]) == 0
    whole = RunReader(RunPaths.of(tmp_path, "sub").evolution)
    bounce = next(e for e in whole.events if e.kind == "bounce")
    k = max(
        s.index for s in whole.snapshots if s.xi < bounce.payload["peak_xi"] + 0.3
    )  # past the peak, before the hold
    assert main(["restart", "sub", "again", "--snapshot", str(k), "--dir", str(tmp_path)]) == 0
    again = RunReader(RunPaths.of(tmp_path, "again").evolution)
    rebounce = next(e for e in again.events if e.kind == "bounce")
    assert rebounce.payload == bounce.payload
    assert rebounce.xi == bounce.xi
    end, reend = whole.end, again.end
    assert end is not None
    assert reend is not None
    assert (reend.xi, reend.payload["reason"]) == (end.xi, "the core bounced")


def test_a_milestones_run_writes_the_initial_state_the_formation_and_the_end_and_steps_free(tmp_path: Path):
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        "grid: {N: 100, Rtilde_max: 12.0, scale: 3.0}\noutput: {snapshots: milestones}\n"
        "excision: {enabled: false}\nevolution: {xi_end: 5.0}\n"
    )
    A = 0.515 * math.e / 8.0  # forms at xi ~ 4.8
    initial = ["initial", "gaussian", "dev", "--config", str(config_path), "--A", f"{A:.12g}", "--ell", "2.0"]
    assert main([*initial, "--dir", str(tmp_path)]) == 0
    assert main(["run", str(config_path), "dev", "--dir", str(tmp_path)]) == 0
    reader = RunReader(RunPaths.of(tmp_path, "dev").evolution)
    formation = next(e for e in reader.events if e.kind == "formation")
    assert [s.xi for s in reader.snapshots] == [0.0, formation.xi, 5.0]
    reach = formation.payload["isolation"]  # the apparent horizon at formation, against the boundary acting since 0
    assert reach["r"] == formation.payload["X_AH"]
    assert reach["since"] == 0.0
    assert reach["needed_sound"] == pytest.approx(reach["r"] + (math.exp(formation.xi / 2) - 1) / math.sqrt(3))
    end = reader.end
    assert end is not None
    assert end.payload["isolation"]["r"] == 0.0  # the origin at the end
    assert [str(v) for v in reader.steps["limit"]].count("output_clip") <= 1  # the end, at most


def test_a_run_that_writes_no_snapshots_keeps_its_initial_state_and_its_records(tmp_path: Path):
    config_path = tmp_path / "bare.yaml"
    config_path.write_text(
        "grid: {N: 40, Rtilde_max: 12.0, scale: 3.0}\noutput: {snapshots: none}\nevolution: {xi_end: 0.3}\n"
    )
    initial = ["initial", "gaussian", "bare", "--config", str(config_path), "--A", "0.05", "--ell", "2.0"]
    assert main([*initial, "--dir", str(tmp_path)]) == 0
    assert main(["run", str(config_path), "bare", "--dir", str(tmp_path)]) == 0
    reader = RunReader(RunPaths.of(tmp_path, "bare").evolution)
    assert [s.xi for s in reader.snapshots] == [0.0]  # the initial state, and nothing after it
    assert len(reader.steps["xi"]) > 0  # the step record is written as always
    assert main(["restart", "bare", "again", "--dir", str(tmp_path)]) == 0  # the run starts again from its start
    assert RunReader(RunPaths.of(tmp_path, "again").evolution).end is not None

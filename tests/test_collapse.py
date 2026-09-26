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

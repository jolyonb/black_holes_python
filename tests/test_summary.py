"""Tests of pbh.summary: what a run says about its hole, recomputed from a synthetic evolution file of known physics."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pbh.causal import isolation
from pbh.cli import main
from pbh.config import EvolutionConfig, GridConfig, MapFamily, RunConfig
from pbh.driver import RunPaths
from pbh.horizon import Horizon, HorizonReport, HorizonRow, NearZone
from pbh.layout import Layout
from pbh.michel import hole_mass_tilde
from pbh.monitors import MonitoredStep
from pbh.output import RunReader, RunWriter
from pbh.state import State
from pbh.summary import RunSummary, as_json, describe, epoch_series, slope, spheres, summarise

N = 40
CONFIG = RunConfig(grid=GridConfig(N=N, Rtilde_max=8.0, map=MapFamily.UNIFORM), evolution=EvolutionConfig(xi_end=8.0))
EOS = CONFIG.fluid.build()
M_INF, EFFICIENCY, XI_FORM, XI_SECOND = 3.0, 1.2, 5.0, 7.9
NAN = float("nan")


def hole(xi: float) -> float:
    """The accretion law at `EFFICIENCY` times Michel's, eq:exc:ahlaw."""
    return M_INF / (1.0 + EFFICIENCY * EOS.accretion_eigenvalue * M_INF * math.exp(-xi))


def row(step: int, xi: float, M_AH: float) -> HorizonRow:
    X = 2.0 * M_AH * math.exp(-float(EOS.alpha) * xi)  # the label radius of 2 M_AH
    sphere = Horizon(j=1, x=X / 8.0, X=X, outer=True)
    report = HorizonReport(np.ones(N + 1), 1, (sphere,), sphere, M_AH, NAN, NAN, 0, NAN, 0, False)
    return HorizonRow.of(step, xi, report, None, NearZone(*[NAN] * 8))


def synthetic_run(directory: Path, flagged: bool, engulfed: bool = True) -> Path:
    """A run whose hole accretes steadily from `XI_FORM`, over an FRW background, with (if `engulfed`) a larger trapped
    region engulfing it at `XI_SECOND`: horizon rows every 0.01, snapshots every 0.1 carrying the hole's mass in the
    first cell, a quoted reading, and the epoch events."""
    paths = RunPaths.of(directory, "synthetic")
    layout = Layout(N)
    with RunWriter(paths.evolution, CONFIG, N, row_type=MonitoredStep) as out:
        out.event(
            0, XI_FORM, "formation", {"M_AH": hole(XI_FORM), "isolation": isolation(EOS, 0.0, XI_FORM, 0.9, 12.0)}
        )
        times = XI_FORM + 0.01 * np.arange(301)
        for step, xi in enumerate(times):
            xi = float(xi)
            second = engulfed and xi >= XI_SECOND
            if second and abs(xi - XI_SECOND) < 1e-9:
                out.event(step, xi, "epoch", {"M_AH": 10.0})
            out.horizon_row(row(step, xi, 10.0 * math.exp(0.01 * (xi - XI_SECOND)) if second else hole(xi)))
            if step % 10 == 0:
                delta_E = np.zeros(N)
                delta_E[0] = hole_mass_tilde(hole(xi), xi, EOS) / 3.0  # the hole's mass, as the cumulative sum counts
                dy = layout.pack(State(E=delta_E, U=np.zeros(N + 1), W=0.0))
                out.snapshot(step, xi, layout, dy, XI_FORM, ())
        quoted = {"epoch_start": XI_FORM, "xi_reading": 7.0, "M_est": 3.0, "bar": 1e-3, "lambda_c_eps": 0.05}
        reach = isolation(EOS, 0.0, 7.0, 0.5, 12.0)
        out.event(
            300, 8.0, "readout", {**quoted, "efficiency": EFFICIENCY, "efficiency_flag": flagged, "isolation": reach}
        )
        out.close(300, 8.0, "completed", isolation=isolation(EOS, 0.0, 8.0, 0.0, 12.0))
    return paths.evolution


def test_the_summary_recovers_the_hole_the_file_was_made_from(tmp_path: Path):
    reader = RunReader(synthetic_run(tmp_path, flagged=False, engulfed=False))
    summary = summarise(reader)
    assert summary.core is None  # the synthetic file has no steps before formation
    assert summary.fold is None  # nor any excised step
    (first,) = summary.epochs
    assert first.xi_start == XI_FORM
    assert first.engulfed is None
    assert first.quoted is not None
    assert first.quoted["M_est"] == 3.0
    ref = first.reference
    assert ref is not None
    assert ref.M_ref == pytest.approx(M_INF, rel=1e-4)
    assert ref.M_AH_end == hole(float((XI_FORM + 0.01 * np.arange(301))[-1]))
    assert ref.efficiency_end == pytest.approx(EFFICIENCY, rel=1e-2)
    assert abs(ref.Q_slope) < 1e-2  # constant efficiency: Q is flat
    assert summary.formation is not None
    assert summary.formation["r"] == 0.9
    assert summary.end is not None
    assert summary.end["r"] == 0.0
    assert summary.status == "completed"
    assert as_json(summary)["status"] == "completed"
    text = describe(summary)
    assert "outer boundary at Rtilde_max = 12, acting since xi = 0:" in text
    assert "at formation (apparent horizon), r = 0.9: needs Rtilde_max >= " in text
    # 0.5 + (e^3.5 - 1) / sqrt 3 and 0.5 + e^3.5 - 1 at the reading; (e^4 - 1) / sqrt 3 and e^4 - 1 at the end
    assert "at reading 0 (apparent horizon), r = 0.5: needs Rtilde_max >= 19 (NOT clear) on sound, 32.6" in text
    assert "at the end (origin), r = 0: needs Rtilde_max >= 30.9 (NOT clear) on sound, 53.6 (NOT clear)" in text
    assert as_json(summary)["end"] == summary.end
    assert ref.bar_below[0.01] == pytest.approx(2.0, abs=0.01)  # the floor decides
    fit = first.fit
    assert fit is not None
    # the law itself, so the free fit recovers it but for the linear resampling between rows 0.01 apart
    assert fit.M_inf == pytest.approx(M_INF, rel=1e-4)
    assert fit.efficiency == pytest.approx(EFFICIENCY, rel=1e-4)
    assert fit.F == pytest.approx(4.0 / 3.0 * EFFICIENCY * EOS.accretion_eigenvalue / 4.0, rel=1e-4)
    # the enclosed mass is the hole plus the background gas, (1/2) e^(-2 xi) R^3: a minimum where the two rates balance
    reached = [s for s in first.spheres if s.minimum_reached]
    assert len(reached) >= 3
    for s in reached:
        expected = min(hole(xi) + 0.5 * math.exp(-2.0 * xi) * s.R**3 for xi in XI_FORM + 0.1 * np.arange(30))
        assert s.m_min == pytest.approx(expected, rel=1e-12)
        assert s.m_min < M_INF
        assert s.deficit == 54.0 / (2.0 * s.k) ** 3
    assert first.extrapolated is not None
    assert first.extrapolated == pytest.approx(M_INF, rel=0.05)


def test_an_engulfed_epoch_keeps_its_reading_and_says_when_it_was_engulfed(tmp_path: Path):
    """The hole engulfed at `XI_SECOND` keeps the reading it quoted, and has no reference, fit or ladder: what engulfs
    it feeds it first. The engulfing epoch is too short for anything but its start."""
    summary = summarise(RunReader(synthetic_run(tmp_path, flagged=False)))
    first, second = summary.epochs
    assert first.engulfed == XI_SECOND
    assert first.quoted is not None
    assert first.quoted["M_est"] == 3.0
    assert first.readings.xi.size > 0
    assert (first.reference, first.fit, first.spheres, first.extrapolated) == (None, None, (), None)
    assert "engulfed at xi = 7.9000 by a trapped region outside it" in describe(summary)
    assert as_json(summary)["epochs"][0]["engulfed"] == XI_SECOND
    assert second.xi_start == XI_SECOND
    assert second.engulfed is None
    assert (second.quoted, second.reference, second.fit, second.spheres) == (None, None, None, ())


def test_the_summary_prints_exports_and_says_when_nothing_formed(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    synthetic_run(tmp_path, flagged=True)
    export = tmp_path / "synthetic.summary.json"
    assert main(["summary", "synthetic", "--dir", str(tmp_path), "--export", str(export)]) == 0
    printed = capsys.readouterr().out
    assert "epoch 0: formed at xi = 5.0000" in printed
    assert "FLAGGED" in printed
    assert "no reading quoted" in printed  # the second epoch
    assert "engulfed at xi = 7.9000" in printed
    data = json.loads(export.read_text())
    epochs = data["epochs"]
    assert epochs[0]["engulfed"] == XI_SECOND
    assert len(epochs[0]["series"]["M_est"]) == len(epochs[0]["series"]["xi"])
    assert epochs[1]["reference"] is None
    assert data["core"] is None
    assert data["fold"] is None
    assert "fold monitor" not in printed
    assert describe(RunSummary(None, [])) == "core: no steps before formation\nno horizon formed"


def test_a_ladder_off_the_grid_or_without_snapshots_gives_no_spheres(tmp_path: Path):
    reader = RunReader(synthetic_run(tmp_path, flagged=False))
    series = (np.array([XI_FORM, 6.0]), np.array([1.0, 1.1]))
    assert spheres(reader, XI_FORM, 6.0, 1e6, series, EOS) == ((), None)  # radii far beyond the outer face
    assert spheres(reader, 7.05, 7.25, M_INF, series, EOS) == ((), None)  # two snapshots in the window: too few
    assert math.isnan(slope(np.array([1.0, 2.0]), np.array([1.0, 2.0])))


def test_repeated_times_are_dropped_as_the_readout_drops_them_and_an_empty_epoch_is_kept(tmp_path: Path):
    """In a deep void the compensated clock can advance by less than the spacing of xi, so the horizon table repeats a
    time: the summary keeps the first row at each time, as `Epoch.add` does. A run interrupted right after its
    formation event has an epoch with no rows: kept, with nothing to read."""
    formed = {"M_AH": hole(XI_FORM), "isolation": isolation(EOS, 0.0, XI_FORM, 0.9, 12.0)}
    times = [float(xi) for xi in XI_FORM + 0.01 * np.arange(301)]
    paths = RunPaths.of(tmp_path, "repeats")
    with RunWriter(paths.evolution, CONFIG, N, row_type=MonitoredStep) as out:
        out.event(0, XI_FORM, "formation", formed)
        for step, xi in enumerate(times):
            out.horizon_row(row(step, xi, hole(xi)))
            if 100 <= step < 110:  # ten times written three times, the copies wrong
                out.horizon_row(row(step, xi, 2.0 * hole(xi)))
                out.horizon_row(row(step, xi, 3.0 * hole(xi)))
    ((start, xi, M_AH),) = epoch_series(RunReader(paths.evolution))
    assert start == XI_FORM
    assert np.array_equal(xi, times)
    assert np.array_equal(M_AH, [hole(x) for x in times])
    (epoch,) = summarise(RunReader(paths.evolution)).epochs
    assert epoch.reference is not None

    interrupted = RunPaths.of(tmp_path, "interrupted")
    with RunWriter(interrupted.evolution, CONFIG, N, row_type=MonitoredStep) as out:
        out.event(0, XI_FORM, "formation", formed)
    summary = summarise(RunReader(interrupted.evolution))
    (empty,) = summary.epochs
    assert empty.xi_start == XI_FORM
    assert empty.readings.xi.size == 0
    assert (empty.quoted, empty.reference, empty.fit, empty.spheres) == (None, None, None, ())

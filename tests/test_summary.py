"""Tests of pbh.summary: what a run says about its hole, recomputed from a synthetic evolution file of known physics."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pbh.cli import main
from pbh.config import EvolutionConfig, GridConfig, MapFamily, RunConfig
from pbh.driver import RunPaths
from pbh.horizon import Horizon, HorizonReport, HorizonRow, NearZone
from pbh.layout import Layout
from pbh.michel import hole_mass_tilde
from pbh.monitors import MonitoredStep
from pbh.output import RunReader, RunWriter
from pbh.state import State
from pbh.summary import describe, slope, spheres, summarise

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


def synthetic_run(directory: Path, flagged: bool) -> Path:
    """A run whose hole accretes steadily from `XI_FORM`, over an FRW background, with a larger trapped region
    engulfing it at `XI_SECOND`: horizon rows every 0.01, snapshots every 0.1 carrying the hole's mass in the first
    cell, a quoted reading, and the two epoch events."""
    paths = RunPaths.of(directory, "synthetic")
    layout = Layout(N)
    with RunWriter(paths.evolution, CONFIG, N, row_type=MonitoredStep) as out:
        out.event(0, XI_FORM, "formation", {"M_AH": hole(XI_FORM)})
        times = XI_FORM + 0.01 * np.arange(301)
        for step, xi in enumerate(times):
            xi = float(xi)
            engulfed = xi >= XI_SECOND
            if engulfed and abs(xi - XI_SECOND) < 1e-9:
                out.event(step, xi, "epoch", {"M_AH": 10.0})
            out.horizon_row(row(step, xi, 10.0 * math.exp(0.01 * (xi - XI_SECOND)) if engulfed else hole(xi)))
            if step % 10 == 0:
                delta_E = np.zeros(N)
                delta_E[0] = hole_mass_tilde(hole(xi), xi, EOS) / 3.0  # the hole's mass, as the cumulative sum counts
                dy = layout.pack(State(E=delta_E, U=np.zeros(N + 1), W=0.0))
                out.snapshot(step, xi, layout, dy, XI_FORM, ())
        quoted = {"epoch_start": XI_FORM, "xi_reading": 7.0, "M_est": 3.0, "bar": 1e-3, "lambda_c_eps": 0.05}
        out.event(300, 8.0, "readout", {**quoted, "efficiency": EFFICIENCY, "efficiency_flag": flagged})
        out.close(300, 8.0, "completed")
    return paths.evolution


def test_the_summary_recovers_the_hole_the_file_was_made_from(tmp_path: Path):
    reader = RunReader(synthetic_run(tmp_path, flagged=False))
    first, second = summarise(reader)
    assert first.xi_start == XI_FORM
    assert first.quoted is not None
    assert first.quoted["M_est"] == 3.0
    ref = first.reference
    assert ref is not None
    assert ref.M_ref == pytest.approx(M_INF, rel=1e-4)
    assert ref.M_AH_end == hole(7.89)
    assert ref.efficiency_end == pytest.approx(EFFICIENCY, rel=1e-2)
    assert abs(ref.Q_slope) < 1e-2  # constant efficiency: Q is flat
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
    # the engulfed epoch is too short for anything but its start
    assert second.xi_start == XI_SECOND
    assert (second.quoted, second.reference, second.fit, second.spheres) == (None, None, None, ())


def test_the_summary_prints_exports_and_says_when_nothing_formed(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    synthetic_run(tmp_path, flagged=True)
    export = tmp_path / "synthetic.summary.json"
    assert main(["summary", "synthetic", "--dir", str(tmp_path), "--export", str(export)]) == 0
    printed = capsys.readouterr().out
    assert "epoch 0: formed at xi = 5.0000" in printed
    assert "FLAGGED" in printed
    assert "no reading quoted" in printed  # the second epoch
    assert "extrapolated" in printed
    data = json.loads(export.read_text())
    assert data[0]["fit"]["efficiency"] == pytest.approx(EFFICIENCY, rel=1e-4)
    assert len(data[0]["series"]["M_est"]) == len(data[0]["series"]["xi"])
    assert data[1]["reference"] is None
    assert describe([]) == "no horizon formed"


def test_a_ladder_off_the_grid_or_without_snapshots_gives_no_spheres(tmp_path: Path):
    reader = RunReader(synthetic_run(tmp_path, flagged=False))
    series = (np.array([XI_FORM, 6.0]), np.array([1.0, 1.1]))
    assert spheres(reader, XI_FORM, 6.0, 1e6, series, EOS) == ((), None)  # radii far beyond the outer face
    assert spheres(reader, 7.05, 7.25, M_INF, series, EOS) == ((), None)  # two snapshots in the window: too few
    assert math.isnan(slope(np.array([1.0, 2.0]), np.array([1.0, 2.0])))

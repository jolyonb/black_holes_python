"""Tests of pbh.pair: the analysis of a pair of runs at N and N/2, on synthetic summaries, and pairs through the CLI."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from pbh import pair
from pbh.cli import main
from pbh.collapse import CollapseHistory
from pbh.config import EvolutionConfig, GridConfig, MapFamily, RunConfig, load, save
from pbh.monitors import MonitoredStep
from pbh.output import RunWriter
from pbh.readout import Readings
from pbh.records import StateRecord, read_initial, write_initial
from pbh.summary import CoreSummary, EpochSummary, RunSummary

NAN = math.nan
EMPTY = Readings(*[np.zeros(0)] * 9)


def run_summary(
    outcome: str = "formed",
    status: str | None = "completed",
    M_est: float | None = None,
    rho_max: float = NAN,
    bounce: float | None = None,
    formation: float | None = None,
    core: bool = True,
) -> RunSummary:
    """A summary with only what the pair reads: the status, the core's outcome and history, and the epochs."""
    history = CollapseHistory(1.0, rho_max, rho_max, 0.5, 1.0, bounce)
    quoted = None if M_est is None else {"M_est": M_est, "bar": 0.01, "systematic": 0.001}
    epochs = [] if formation is None else [EpochSummary(formation, EMPTY, quoted, None, None, (), None)]
    return RunSummary(CoreSummary(outcome, history, None) if core else None, epochs, status=status)


def test_the_half_configuration_halves_n_and_changes_nothing_else():
    config = load_yaml("grid: {N: 40, Rtilde_max: 12.0, scale: 3.0}\nevolution: {xi_end: 0.2}\n")
    half = pair.half_config(config)
    assert half.grid.N == 20
    assert half.model_dump(exclude={"grid"}) == config.model_dump(exclude={"grid"})
    assert half.grid.model_dump(exclude={"N"}) == config.grid.model_dump(exclude={"N"})
    grid = GridConfig(N=8, Rtilde_max=8.0, map=MapFamily.UNIFORM)
    uniform = RunConfig(grid=grid, evolution=EvolutionConfig(xi_end=1.0))
    assert pair.half_config(uniform).grid.map is MapFamily.UNIFORM
    for N in (41, 2):
        odd = uniform.model_copy(update={"grid": GridConfig(N=N, Rtilde_max=8.0, map=MapFamily.UNIFORM)})
        with pytest.raises(ValueError, match=f"even grid.N of at least 4 to halve, not {N}"):
            pair.half_config(odd)
    assert pair.half_name("g") == "g.half"


def load_yaml(text: str) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(text))


def test_the_outcome_is_aborted_unless_completed_and_formed_after_a_restart_past_formation():
    assert pair.outcome(run_summary("formed", status="aborted")) == "aborted"
    assert pair.outcome(run_summary("bounced", status="interrupted")) == "aborted"
    assert pair.outcome(run_summary("bounced")) == "bounced"
    assert pair.outcome(run_summary(core=False, formation=3.0)) == "formed"  # no step before formation
    assert pair.outcome(run_summary(core=False)) == "undecided"


def test_the_mass_and_the_peak_carry_a_third_of_the_difference_and_the_times_none(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(pair, "F_COVERAGE", 2.0)
    full = run_summary(M_est=10.0, rho_max=100.0, formation=4.0)
    half = run_summary(M_est=10.3, rho_max=97.0, formation=4.06)
    by_name = {e.name: e for e in pair.estimates(full, half)}
    assert tuple(by_name) == pair.OBSERVABLES
    assert (by_name["formation_xi"].value, by_name["formation_xi"].half) == (4.0, 4.06)
    assert math.isnan(by_name["formation_xi"].error)  # quantised to the step: no error from the pair
    assert by_name["M_est"].error == pytest.approx(2.0 * -0.1)  # F on the mass only
    assert by_name["rho_max"].error == pytest.approx(1.0)
    assert (by_name["M_est"].value, by_name["M_est"].half) == (10.0, 10.3)
    assert math.isnan(by_name["bounce_xi"].error)  # neither bounced
    bounced = pair.observables(run_summary("bounced", rho_max=50.0, bounce=6.5))
    assert (bounced["bounce_xi"], bounced["rho_max"]) == (6.5, 50.0)
    assert math.isnan(bounced["M_est"])
    assert math.isnan(bounced["formation_xi"])
    assert math.isnan(pair.observables(run_summary(core=False))["rho_max"])


def test_the_normal_distribution_function():
    assert pair.normal_cdf(0.0) == 0.5
    assert pair.normal_cdf(1.6448536269514722) == pytest.approx(0.95, abs=1e-12)
    assert pair.normal_cdf(math.inf) == 1.0


def test_above_threshold_the_verdict_is_read_from_the_mass():
    near = pair.trust(run_summary(M_est=1.0, formation=4.0), run_summary(M_est=2.0, formation=4.0), 400)
    x = 1.0  # |dM/M| of the run at N: 3 GAMMA / (K x) = 0.84 standard deviations
    assert (near.side, near.x) == ("above", x)
    assert near.probability == pytest.approx(pair.normal_cdf(3.0 * pair.GAMMA / (pair.K_TRUST * x)))
    assert near.verdict == "untrusted"
    needed = 400.0 * math.sqrt(1.6448536269514722 * pair.K_TRUST * x / (3.0 * pair.GAMMA))
    assert near.N_needed == 2 * math.ceil(needed / 2.0)  # rounded up to an even N
    assert needed > 400
    far = pair.trust(run_summary(M_est=10.0, formation=4.0), run_summary(M_est=10.0001, formation=4.0), 400)
    assert far.verdict == "trusted"
    assert far.probability == 1.0  # far from threshold x is tiny and P -> 1
    assert far.N_needed is not None
    assert far.N_needed < 400  # already more than enough
    exact = pair.trust(run_summary(M_est=10.0, formation=4.0), run_summary(M_est=10.0, formation=4.0), 400)
    assert (exact.probability, exact.N_needed) == (1.0, 2)


def test_below_threshold_the_verdict_is_read_from_the_peak_density():
    t = pair.trust(run_summary("bounced", rho_max=100.0), run_summary("bounced", rho_max=100.0 * math.exp(-0.2)), 800)
    assert t.side == "below"
    assert t.x == pytest.approx(0.1)  # |d ln rho_max| / 2
    assert t.probability == pytest.approx(pair.normal_cdf(3.0 * pair.GAMMA / (pair.K_TRUST * 0.1)))
    assert t.verdict == "trusted"


def test_outcomes_that_differ_or_abort_are_untrusted_and_agreement_without_a_measure_gives_no_verdict():
    differ = pair.trust(run_summary(M_est=1.0, formation=4.0), run_summary("bounced", rho_max=10.0), 400)
    assert (differ.verdict, differ.side, differ.probability, differ.N_needed) == ("untrusted", None, 0.0, None)
    aborted = pair.trust(run_summary(M_est=1.0, formation=4.0), run_summary(status="aborted", formation=4.0), 400)
    assert aborted.probability == 0.0
    both = pair.trust(run_summary(status="aborted"), run_summary(status="aborted"), 400)
    assert (both.verdict, both.side, both.probability) == ("untrusted", None, 0.0)  # an abort is no outcome
    for full, half in [
        (run_summary("undecided"), run_summary("undecided")),
        (run_summary(formation=4.0), run_summary(M_est=1.0, formation=4.0)),  # formed, but one read no mass
        (run_summary("bounced"), run_summary("bounced")),  # no peak (cannot happen with a bounce, but guarded)
    ]:
        none = pair.trust(full, half, 400)
        assert (none.verdict, none.side, none.probability, none.N_needed) == ("no verdict", None, None, None)
        assert math.isnan(none.x)


def test_the_mass_budget_adds_its_components_in_quadrature(monkeypatch: pytest.MonkeyPatch):
    full, half = run_summary(M_est=10.0, formation=4.0), run_summary(M_est=10.6, formation=4.0)
    b = pair.mass_budget(full, half)
    assert b is not None
    spatial = pair.F_COVERAGE * 0.2
    assert (b.M_est, b.spatial, b.readout, b.window) == (10.0, pytest.approx(spatial), 0.1, 0.01)
    assert b.systematics == {name: pytest.approx(10.0 * f) for name, f in pair.SYSTEMATICS.items()}
    quoted = sum((10.0 * f) ** 2 for f in pair.SYSTEMATICS.values())
    assert b.total == pytest.approx(math.sqrt(spatial**2 + 0.1**2 + 0.01**2 + quoted))
    monkeypatch.setattr(pair, "SYSTEMATICS", {"time_step": 0.003, "viscosity": 0.004})
    monkeypatch.setattr(pair, "F_COVERAGE", 1.5)
    b = pair.mass_budget(full, half)
    assert b is not None
    assert b.systematics == {"time_step": pytest.approx(0.03), "viscosity": pytest.approx(0.04)}
    assert b.spatial == pytest.approx(0.3)
    assert b.total == pytest.approx(math.sqrt(0.3**2 + 0.1**2 + 0.01**2 + 0.03**2 + 0.04**2))
    assert pair.mass_budget(run_summary(formation=4.0), half) is None  # the run at N read no mass
    assert pair.mass_budget(full, run_summary(formation=4.0)) is None  # its companion did not
    assert pair.mass_budget(run_summary("undecided"), run_summary("undecided")) is None


def calibrated(N: int) -> RunConfig:
    """A configuration at the settings of K_TRUST's calibration, so that the verdict carries no caveat."""
    return load_yaml(
        f"grid: {{N: {N}, Rtilde_max: 30.0, scale: 3.0}}\nstepping: {{cap_tolerance: 1.0e-7}}\n"
        "evolution: {xi_start: -10.0, xi_end: 8.0}\n"
    )


def test_the_pair_is_described_and_exported():
    full, half = run_summary(M_est=10.0, formation=4.0), run_summary(M_est=10.03, formation=4.01)
    trusted = pair.analyse(full, half, calibrated(400))
    assert trusted.outcomes == ("formed", "formed")
    assert trusted.agree
    assert trusted.caveats == ()
    text = "\n".join(pair.describe(trusted))
    assert "pair: N = 400 formed, N/2 = 200 formed (agree)" in text
    assert "threshold: trusted above, P = 1.000 (x = 0.003); 95% needs N >= " in text
    assert "caveat" not in text
    assert "M_est: 10 (N/2: 10.03), spatial error -0.0153 (F = 1.53)" in text
    assert "formation_xi: 4 (N/2: 4.01), sampled at the steps: no error estimate" in text
    assert "bounce_xi" not in text  # absent at both
    assert "M_est = 10 +- 0.1 R_H (spatial 0.015, readout 0.1, window 0.01, time_step 0.0011, viscosity 0, " in text
    assert f"    {pair.BUDGET_CAVEAT}" in text  # the total does not cover the systematics' growth near threshold
    data = pair.as_json(trusted)
    assert data["trust"]["verdict"] == "trusted"
    assert data["budget"]["readout"] == 0.1
    assert data["estimates"]["M_est"]["half"] == 10.03
    assert data["caveats"] == []
    json.dumps(data)
    differ = pair.analyse(run_summary(M_est=1.0, formation=4.0), run_summary("bounced", rho_max=10.0), calibrated(400))
    assert not differ.agree
    text = "\n".join(pair.describe(differ))
    assert "(DIFFER)" in text
    assert "threshold: untrusted, the outcomes differ between N and N/2" in text
    assert "M_est = " not in text
    assert pair.as_json(differ)["budget"] is None
    nothing = pair.describe(pair.analyse(run_summary("undecided"), run_summary("undecided"), calibrated(40)))
    assert "  threshold: no verdict (neither a read mass nor a bounce to measure the distance)" in nothing
    bounced = pair.describe(
        pair.analyse(run_summary("bounced", rho_max=100.0), run_summary("bounced", rho_max=97.0), calibrated(400))
    )
    assert "  rho_max: 100 (N/2: 97), spatial error +1 (uncalibrated)" in bounced
    one = pair.analyse(run_summary(M_est=1.0, formation=4.0), run_summary(status="aborted"), calibrated(400))
    assert not one.agree
    assert "  threshold: untrusted, the run at N/2 aborted" in pair.describe(one)
    both = pair.analyse(run_summary(status="aborted"), run_summary(status="aborted"), calibrated(400))
    assert not both.agree
    text = "\n".join(pair.describe(both))
    assert "pair: N = 400 aborted, N/2 = 200 aborted (both aborted)" in text
    assert "threshold: untrusted, the run at N and the run at N/2 aborted" in text


def test_a_looser_cap_or_a_later_start_than_the_calibration_s_is_named_in_the_verdict():
    production = load_yaml(  # the former default cap, and a start after the calibration's
        "grid: {N: 400, Rtilde_max: 30.0, scale: 3.0}\nstepping: {cap_tolerance: 1.0e-5}\n"
        "evolution: {xi_start: -6.0, xi_end: 8.0}\n"
    )
    analysed = pair.analyse(run_summary(M_est=10.0, formation=4.0), run_summary(M_est=10.03, formation=4.0), production)
    assert len(analysed.caveats) == 2
    text = "\n".join(pair.describe(analysed))
    assert "    caveat: the step cap's tolerance 1e-05 is looser than the 1e-07 of K's calibration" in text
    assert "    caveat: xi_start = -6 is later than the -10 of K's calibration" in text
    assert pair.caveats(calibrated(400)) == ()


# --- through the command line ---

CONFIG = """
grid: {N: 40, Rtilde_max: 12.0, scale: 3.0}
output: {snapshot_spacing: 0.1, snapshot_spacing_min: 0.1}
evolution: {xi_end: 0.2}
"""


def gaussian(tmp_path: Path, config: Path, *more: str) -> int:
    args = ["initial", "gaussian", "g", "--config", str(config), "--A", "0.05", "--ell", "2.0", "--dir", str(tmp_path)]
    return main([*args, *more])


def test_a_pair_through_the_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    config = tmp_path / "small.yaml"
    config.write_text(CONFIG)
    assert gaussian(tmp_path, config, "--pair") == 0
    assert "g.half.initial.h5" in capsys.readouterr().out
    half_initial = read_initial(tmp_path / "g.half.initial.h5")
    assert half_initial.X.size == 21  # the faces of the grid with N halved
    assert half_initial.provenance["half_of"] == "g"
    assert half_initial.provenance["A"] == 0.05
    assert "half_of" not in read_initial(tmp_path / "g.initial.h5").provenance
    assert main(["run", str(config), "g", "--pair", "--dir", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert printed.count("completed") == 2
    saved: dict[str, Any] = yaml.safe_load((tmp_path / "g.half.config.yaml").read_text())
    assert saved["provenance"]["half_of"] == "g"
    assert saved["grid"]["N"] == 20
    assert "half_of" not in yaml.safe_load((tmp_path / "g.config.yaml").read_text())["provenance"]
    assert load(tmp_path / "g.half.config.yaml") == pair.half_config(load(config))
    export = tmp_path / "g.json"
    assert main(["summary", "g", "--dir", str(tmp_path), "--export", str(export)]) == 0
    printed = capsys.readouterr().out
    assert "pair: N = 40 undecided, N/2 = 20 undecided (agree)" in printed
    assert "caveat: xi_start = 0 is later than" in printed  # the test's configuration is not the calibration's
    data = json.loads(export.read_text())
    assert data["pair"]["outcomes"] == ["undecided", "undecided"]
    assert data["status"] == "completed"
    assert main(["summary", "g.half", "--dir", str(tmp_path)]) == 0  # the companion is a run like any other
    assert "pair:" not in capsys.readouterr().out


def test_a_pair_with_one_run_unfinished_summarises_the_finished_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    config = tmp_path / "small.yaml"
    config.write_text(CONFIG)
    assert gaussian(tmp_path, config) == 0
    assert main(["run", str(config), "g", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    parsed = load(config)
    half = pair.half_config(parsed)
    save(half, tmp_path / "g.half.config.yaml", half_of="g")
    with RunWriter(tmp_path / "g.half.evolution.h5", half, 20, row_type=MonitoredStep) as running:
        running.flush()
        export = tmp_path / "g.json"
        assert main(["summary", "g", "--dir", str(tmp_path), "--export", str(export)]) == 0
        printed = capsys.readouterr().out
        assert "core: undecided" in printed
        assert "pair: g.half has not ended; no pair analysis yet" in printed
        assert "pair" not in json.loads(export.read_text())
    # the other way round: the run at N still going, its companion started after it and finished
    with RunWriter(tmp_path / "g.evolution.h5", parsed, 40, row_type=MonitoredStep) as also_running:
        also_running.flush()
        with RunWriter(tmp_path / "g.half.evolution.h5", half, 20, row_type=MonitoredStep):
            pass
        assert main(["summary", "g", "--dir", str(tmp_path)]) == 0
        assert "pair: g has not ended; no pair analysis yet" in capsys.readouterr().out


def test_an_odd_n_or_a_missing_companion_is_refused_before_anything_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    odd = tmp_path / "odd.yaml"
    odd.write_text(CONFIG.replace("N: 40", "N: 41"))
    assert gaussian(tmp_path, odd, "--pair") == 1
    assert "even grid.N of at least 4 to halve, not 41" in capsys.readouterr().err
    assert not (tmp_path / "g.initial.h5").exists()  # refused before anything was written
    config = tmp_path / "small.yaml"
    config.write_text(CONFIG)
    assert gaussian(tmp_path, config) == 0  # no companion's initial file
    assert main(["run", str(config), "g", "--pair", "--dir", str(tmp_path)]) == 1
    assert "g.half.initial.h5" in capsys.readouterr().err
    assert not (tmp_path / "g.evolution.h5").exists()  # the run at N did not start


def test_the_companion_runs_whether_the_first_run_completes_or_aborts_and_an_abort_is_status_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    config = tmp_path / "small.yaml"
    config.write_text(CONFIG)
    parsed = load(config)
    for name, c, provenance in (
        ("broken", parsed, {}),
        ("broken.half", pair.half_config(parsed), {"half_of": "broken"}),
    ):
        geo = c.scheme().frame(0.0).geo
        X = geo.X[: geo.N + 1]
        write_initial(
            tmp_path / f"{name}.initial.h5",
            StateRecord(np.zeros(geo.N), np.zeros(geo.N + 1), NAN, 0.0, X, 0.0, 0, provenance),
        )
    assert main(["run", str(config), "broken", "--pair", "--dir", str(tmp_path)]) == 2
    assert capsys.readouterr().out.count("aborted") == 2
    assert main(["summary", "broken", "--dir", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "pair: N = 40 aborted, N/2 = 20 aborted (both aborted)" in printed
    assert "threshold: untrusted, the run at N and the run at N/2 aborted" in printed


def test_a_half_datum_that_does_not_name_the_run_is_refused_before_anything_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    config = tmp_path / "small.yaml"
    config.write_text(CONFIG)
    assert gaussian(tmp_path, config, "--pair") == 0
    (tmp_path / "g.initial.h5").rename(tmp_path / "h.initial.h5")
    (tmp_path / "g.half.initial.h5").rename(tmp_path / "h.half.initial.h5")  # still says half_of g
    capsys.readouterr()
    assert main(["run", str(config), "h", "--pair", "--dir", str(tmp_path)]) == 1
    assert "h.half.initial.h5 is not the half datum of h" in capsys.readouterr().err
    assert not (tmp_path / "h.evolution.h5").exists()


def test_a_leftover_companion_is_not_paired_with_a_rerun(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    config = tmp_path / "small.yaml"
    config.write_text(CONFIG)
    assert gaussian(tmp_path, config, "--pair") == 0
    assert main(["run", str(config), "g", "--pair", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    stale = "pair: g.half is not this run's companion; no pair analysis"
    # the same configuration run again without its companion: the companion is older than the run
    assert main(["run", str(config), "g", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["summary", "g", "--dir", str(tmp_path)]) == 0
    assert stale in capsys.readouterr().out
    # the run at another N over the leftover companion
    other = tmp_path / "other.yaml"
    other.write_text(CONFIG.replace("N: 40", "N: 60"))
    assert gaussian(tmp_path, other) == 0
    assert main(["run", str(other), "g", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["summary", "g", "--dir", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert stale in printed
    assert "N/2 = 30" not in printed
    # and a companion whose saved configuration is gone cannot be confirmed
    (tmp_path / "g.half.config.yaml").unlink()
    assert main(["summary", "g", "--dir", str(tmp_path)]) == 0
    assert stale in capsys.readouterr().out

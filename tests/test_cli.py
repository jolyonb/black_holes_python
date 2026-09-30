"""Tests of pbh.cli: the four verbs, in process."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

import pbh
from pbh.cli import main
from pbh.eos import RADIATION, Background, EquationOfState
from pbh.initial import GrowingMode
from pbh.output import RunReader
from pbh.profiles import Gaussian
from pbh.records import read_initial
from pbh.timestep import Engine
from pbh.types import FloatArray

CONFIG = """
grid: {N: 40, Rtilde_max: 12.0, scale: 3.0}
output: {snapshot_spacing: 0.1, snapshot_spacing_min: 0.1}
evolution: {xi_end: 0.2}
"""


def write_config(tmp_path: Path) -> Path:
    path = tmp_path / "small.yaml"
    path.write_text(CONFIG)
    return path


def test_validate_prints_the_complete_configuration(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["validate", str(write_config(tmp_path))]) == 0
    printed = yaml.safe_load(capsys.readouterr().out)
    assert printed["grid"] == {"N": 40, "Rtilde_max": 12.0, "map": "sinh", "scale": 3.0}
    assert printed["stepping"]["courant_number"] == 0.75


def test_a_bad_configuration_is_an_error_message_and_a_nonzero_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    path = tmp_path / "bad.yaml"
    path.write_text("grid: {N: 40, Rtilde_max: 12.0, scale: 3.0}\nevolution: {xi_end: 0.2}\ncolour: blue\n")
    assert main(["validate", str(path)]) == 1
    assert "colour" in capsys.readouterr().err


def test_initial_gaussian_then_run_then_restart(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    config = write_config(tmp_path)
    assert (
        main(
            ["initial", "gaussian", "g", "--config", str(config), "--A", "0.05", "--ell", "2.0", "--dir", str(tmp_path)]
        )
        == 0
    )
    assert "peak compaction 0.1472" in capsys.readouterr().out
    initial = read_initial(tmp_path / "g.initial.h5")
    assert initial.provenance["profile"] == "gaussian"
    assert initial.provenance["A"] == 0.05
    assert 0.0 < initial.provenance["correction_ratio"] < 0.05
    assert main(["run", str(config), "g", "--dir", str(tmp_path)]) == 0
    assert "completed" in capsys.readouterr().out
    reader = RunReader(tmp_path / "g.evolution.h5")
    assert [s.xi for s in reader.snapshots] == [0.0, 0.1, 0.2]
    assert reader.engine is Engine.PYTHON  # the default, recorded in both files
    assert yaml.safe_load((tmp_path / "g.config.yaml").read_text())["provenance"]["engine"] == "python"
    assert main(["restart", "g", "h", "--dir", str(tmp_path), "--snapshot", "1", "--engine", "python"]) == 0
    assert "completed" in capsys.readouterr().out
    again = read_initial(tmp_path / "h.initial.h5")
    assert again.xi == 0.1
    continued = RunReader(tmp_path / "h.evolution.h5")
    end = continued.end
    assert end is not None
    assert end.payload["isolation"]["since"] == 0.0  # the source's start, from its configuration, not the snapshot's
    assert again.provenance["source"] == "g.evolution.h5"
    assert [s.xi for s in RunReader(tmp_path / "h.evolution.h5").snapshots] == [0.1, 0.2]


def test_an_engine_that_is_not_installed_or_does_not_exist_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    config = write_config(tmp_path)
    args = ["initial", "gaussian", "g", "--config", str(config), "--A", "0.05", "--ell", "2.0", "--dir", str(tmp_path)]
    assert main(args) == 0
    monkeypatch.delitem(sys.modules, "pbh.rust_engine", raising=False)  # as on a machine without the Rust engine
    monkeypatch.delattr(pbh, "rust_engine", raising=False)
    monkeypatch.setitem(sys.modules, "pbh_engine", None)
    assert main(["run", str(config), "g", "--engine", "rust", "--dir", str(tmp_path)]) == 1
    assert "pbh_engine is not installed: install it with `uv sync --group rust`" in capsys.readouterr().err
    assert main(["run", str(config), "g", "--engine", "fortran", "--dir", str(tmp_path)]) == 2  # a usage error


def test_a_gaussian_the_reconstruction_refuses_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    config = write_config(tmp_path)
    args = ["initial", "gaussian", "n", "--config", str(config), "--A", "0.05", "--ell", "0.5", "--dir", str(tmp_path)]
    assert main(args) == 1
    assert "cannot be read from one field" in capsys.readouterr().err


def test_a_missing_initial_file_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["run", str(write_config(tmp_path)), "nothing", "--dir", str(tmp_path)]) == 1
    assert "nothing.initial.h5" in capsys.readouterr().err


def test_a_run_whose_initial_data_precede_its_configured_start_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    config = write_config(tmp_path)
    args = ["initial", "gaussian", "g", "--config", str(config), "--A", "0.05", "--ell", "2.0", "--dir", str(tmp_path)]
    assert main(args) == 0
    later = tmp_path / "later.yaml"
    later.write_text(CONFIG.replace("evolution: {xi_end: 0.2}", "evolution: {xi_start: 0.1, xi_end: 0.2}"))
    assert main(["run", str(later), "g", "--dir", str(tmp_path)]) == 1
    assert "before the configuration's xi_start = 0.1" in capsys.readouterr().err


def test_the_gaussian_datum_exists_for_radiation_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    path = tmp_path / "dust.yaml"
    path.write_text("fluid: {w: 1/6}\ngrid: {N: 40, Rtilde_max: 12.0, scale: 3.0}\nevolution: {xi_end: 0.2}\n")
    args = ["initial", "gaussian", "d", "--config", str(path), "--A", "0.05", "--ell", "2.0", "--dir", str(tmp_path)]
    assert main(args) == 1
    assert "radiation only" in capsys.readouterr().err


def test_a_usage_error_is_status_two_and_help_is_status_zero(capsys: pytest.CaptureFixture[str]):
    assert main(["run"]) == 2  # missing arguments: cyclopts prints the usage error
    assert main(["--help"]) == 0
    assert "validate" in capsys.readouterr().out
    assert main([]) == 0  # no command: the help, and a normal return
    assert "validate" in capsys.readouterr().out


def test_an_aborted_run_leaves_with_status_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    import numpy as np

    from pbh.records import StateRecord, write_initial

    config = write_config(tmp_path)
    from pbh.config import load

    geo = load(config).scheme().frame(0.0).geo
    X = geo.X[: geo.N + 1]
    broken = StateRecord(np.zeros(geo.N), np.zeros(geo.N + 1), float("nan"), 0.0, X, 0.0, 0, {})
    write_initial(tmp_path / "broken.initial.h5", broken)
    assert main(["run", str(config), "broken", "--dir", str(tmp_path)]) == 2
    assert "aborted" in capsys.readouterr().out


def test_the_module_can_be_run_directly():
    import subprocess
    import sys

    done = subprocess.run([sys.executable, "-m", "pbh.cli", "--help"], capture_output=True, text=True, check=False)
    assert done.returncode == 0
    assert "validate" in done.stdout


def test_a_perturbation_far_below_round_off_of_the_background_reaches_the_file(tmp_path: Path):
    config = tmp_path / "early.yaml"
    config.write_text(CONFIG.replace("evolution: {xi_end: 0.2}", "evolution: {xi_start: -30.0, xi_end: -29.0}"))
    A = f"{1e-3 * math.e / 8.0 * math.exp(-30.0)!r}"  # delta_m ~ 3e-17: 1 + delta_m is 1 to the bit
    args = ["initial", "gaussian", "t", "--config", str(config), "--A", A, "--ell", "2.0"]
    assert main([*args, "--dir", str(tmp_path)]) == 0
    initial = read_initial(tmp_path / "t.initial.h5")
    mass = 3.0 * np.cumsum(initial.delta_E) / initial.X[1:] ** 3
    assert 1e-17 < float(np.max(mass)) < 1e-16


WIDE = """
grid: {N: 200, Rtilde_max: 30.0, scale: 3.0}
output: {snapshots: milestones}
evolution: {xi_start: XI_START, xi_end: XI_END}
"""


def run_gaussian(tmp_path: Path, name: str, C: float, xi0: float, xi_end: float) -> None:
    """Write a Gaussian (ell = 2) of peak compaction C at xi0 and run it to xi_end."""
    config = tmp_path / f"{name}.yaml"
    config.write_text(WIDE.replace("XI_START", repr(xi0)).replace("XI_END", repr(xi_end)))
    A = repr(C * math.e / 8.0 * math.exp(xi0))
    args = ["initial", "gaussian", name, "--config", str(config), "--A", A, "--ell", "2.0"]
    assert main([*args, "--dir", str(tmp_path)]) == 0
    assert main(["run", str(config), name, "--dir", str(tmp_path)]) == 0


def last_fields(tmp_path: Path, name: str) -> tuple[float, FloatArray, FloatArray, FloatArray]:
    """The last snapshot's time, radii and relative deviations `(delta_m, delta_U)` at faces 1..N."""
    reader = RunReader(tmp_path / f"{name}.evolution.h5")
    end = reader.snapshot(len(reader.snapshots) - 1)
    X = end.X[1:]
    return end.xi, X, 3.0 * np.cumsum(end.delta_E) / X**3, end.delta_U[1:] / X


def relative_l2(values: FloatArray, reference: FloatArray, X: FloatArray) -> float:
    """`|values - reference| / |reference|` in L2 with the weight `X^2 dX` on the faces `X`."""
    w = X**2 * np.diff(np.concatenate(([0.0], X)))
    return float(np.sqrt(np.sum(w * (values - reference) ** 2) / np.sum(w * reference**2)))


@pytest.mark.slow
def test_a_perturbation_far_below_round_off_evolves_as_linear_theory_says(tmp_path: Path):
    # Compaction 1e-3 given at xi = -30 (delta_m ~ 3e-17) and run to -10 (delta_m ~ 2e-8): the linear evolution is
    # exact up to O(C epsilon^2) ~ 1e-9 there, so what remains is the scheme's truncation error (measured 4.5e-5),
    # where carried through the full state the whole signal was lost.
    run_gaussian(tmp_path, "early", 1e-3, -30.0, -10.0)
    xi, X, delta_m, delta_U = last_fields(tmp_path, "early")
    assert xi == -10.0
    eos = EquationOfState(RADIATION)
    profile = Gaussian(A=1e-3 * math.e / 8.0 * math.exp(-30.0), ell=2.0)
    mode, _ = GrowingMode.from_profile(profile, "m", 30.0, Background.at(eos, -30.0))
    bg = Background.at(eos, xi)
    assert relative_l2(delta_m, mode.delta_m(bg, X), X) < 1e-4
    assert relative_l2(delta_U, mode.delta_U(bg, X), X) < 1e-4

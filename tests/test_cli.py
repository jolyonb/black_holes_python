"""Tests of pbh.cli: the four verbs, in process."""

from pathlib import Path

import pytest
import yaml

from pbh.cli import main
from pbh.output import RunReader
from pbh.records import read_initial

CONFIG = """
grid: {N: 40, Rtilde_max: 12.0, scale: 3.0}
output: {snapshot_spacing: 0.1}
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
    assert main(["restart", "g", "h", "--dir", str(tmp_path), "--snapshot", "1"]) == 0
    assert "completed" in capsys.readouterr().out
    again = read_initial(tmp_path / "h.initial.h5")
    assert again.xi == 0.1
    assert again.provenance["source"] == "g.evolution.h5"
    assert [s.xi for s in RunReader(tmp_path / "h.evolution.h5").snapshots] == [0.1, 0.2]


def test_a_gaussian_the_reconstruction_refuses_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    config = write_config(tmp_path)
    args = ["initial", "gaussian", "n", "--config", str(config), "--A", "0.05", "--ell", "0.5", "--dir", str(tmp_path)]
    assert main(args) == 1
    assert "cannot be read from one field" in capsys.readouterr().err


def test_a_missing_initial_file_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["run", str(write_config(tmp_path)), "nothing", "--dir", str(tmp_path)]) == 1
    assert "nothing.initial.h5" in capsys.readouterr().err


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

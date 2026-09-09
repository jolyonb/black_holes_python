"""Tests for the command line interface."""

from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pbh import cli
from pbh.base import FloatArray
from pbh.cli import build_parser, main
from pbh.initial import growingmode
from pbh.ms import MS

InitialData = tuple[FloatArray, FloatArray, FloatArray]


def test_help_lists_schemes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
    assert "lagrangian" in capsys.readouterr().out


def test_defaults_match_historical_driver() -> None:
    args = build_parser().parse_args([])
    assert args.scheme == "eulerian"
    assert args.gridpoints == 600
    assert args.amplitude == 0.175
    assert args.viscosity == 2.0
    assert args.max_time == 7.0
    assert args.output == "output.dat"


@pytest.mark.parametrize("scheme", ["eulerian", "lagrangian"])
def test_run_writes_output(tmp_path: Path, scheme: str, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "run.dat"
    argv = ["-o", str(out), "--scheme", scheme, "--gridpoints", "100", "--max-time", "0.5", "--quiet"]
    assert main(argv) == 0
    assert "Status: TIMEOUT" in capsys.readouterr().out
    blocks = out.read_text().strip().split("\n\n")
    assert len(blocks) == 6  # xi = 0.0, 0.1, ..., 0.5
    assert all(len(block.splitlines()) == 101 for block in blocks)


def test_run_writes_npz(tmp_path: Path) -> None:
    out = tmp_path / "run.npz"
    assert main(["-o", str(out), "--gridpoints", "100", "--max-time", "0.5", "--quiet"]) == 0
    with np.load(out) as data:
        assert data["xi"] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        assert data["rho"].shape == (6, 100)
        assert data["index"].shape == (100,)


def test_unphysical_run_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "bad.dat"
    argv = ["-o", str(out), "--gridpoints", "100", "--amplitude", "0.25", "--max-time", "0.5", "--quiet"]
    assert main(argv) == 1
    assert "Status: NEGATIVE_GAMMA2" in capsys.readouterr().out


def test_run_from_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first = tmp_path / "first.dat"
    assert main(["-o", str(first), "--gridpoints", "100", "--max-time", "0.2", "--quiet"]) == 0
    second = tmp_path / "second.dat"
    assert main([str(first), "-o", str(second), "--max-time", "0.2", "--quiet"]) == 0
    assert "Status: TIMEOUT" in capsys.readouterr().out
    # Loading from the first block should reproduce the first file exactly
    assert second.read_text() == first.read_text()
    # Resuming from the last block continues from there
    third = tmp_path / "third.npz"
    assert main([str(first), "--snapshot", "-1", "-o", str(third), "--max-time", "0.4", "--quiet"]) == 0
    with np.load(third) as data:
        assert data["xi"] == pytest.approx([0.2, 0.3, 0.4])


def test_w_option_parses_rationals() -> None:
    parser = build_parser()
    assert parser.parse_args([]).w == Fraction(1, 3)
    assert parser.parse_args(["--w", "1/3"]).w == Fraction(1, 3)
    assert parser.parse_args(["--w", "0.2"]).w == Fraction(1, 5)
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--w", "abc"])
    assert excinfo.value.code == 2


def test_w_is_passed_to_driver_and_initial_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--w reaches both the evolver and the growing-mode construction, on a fresh run and on a restart."""
    seen: list[tuple[str, Fraction]] = []

    class RecordingMS(MS):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            seen.append(("MS", kwargs["w"]))
            super().__init__(*args, **kwargs)

    def recording_growingmode(grid: FloatArray, deltam0: FloatArray, w: Fraction) -> InitialData:
        seen.append(("growingmode", w))
        return growingmode(grid, deltam0, w=w)

    monkeypatch.setattr(cli, "MS", RecordingMS)
    monkeypatch.setattr(cli, "growingmode", recording_growingmode)
    first = tmp_path / "first.dat"
    assert main(["-o", str(first), "--gridpoints", "50", "--max-time", "0.1", "--quiet", "--w", "2/6"]) == 0
    assert seen == [("MS", Fraction(1, 3)), ("growingmode", Fraction(1, 3))]
    seen.clear()
    second = tmp_path / "second.dat"
    assert main([str(first), "-o", str(second), "--max-time", "0.1", "--quiet", "--w", "2/6"]) == 0
    assert seen == [("MS", Fraction(1, 3))]


def test_explicit_radiation_w_reproduces_default_output(tmp_path: Path) -> None:
    base_args = ["--gridpoints", "100", "--max-time", "0.2", "--quiet"]
    default, explicit = tmp_path / "default.dat", tmp_path / "explicit.dat"
    assert main(["-o", str(default), *base_args]) == 0
    assert main(["-o", str(explicit), *base_args, "--w", "1/3"]) == 0
    assert explicit.read_bytes() == default.read_bytes()


def test_non_radiation_w_is_refused_before_evolution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI rejects w != 1/3 up front (the outer boundary condition exists only for radiation), on both paths."""
    out = tmp_path / "w02.dat"
    with pytest.raises(SystemExit) as excinfo:
        main(["-o", str(out), "--gridpoints", "50", "--max-time", "0.1", "--quiet", "--w", "0.2"])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "only w = 1/3" in captured.err
    assert "Beginning evolution" not in captured.out
    assert not out.exists()
    # The restart path is guarded too
    first = tmp_path / "first.dat"
    assert main(["-o", str(first), "--gridpoints", "50", "--max-time", "0.1", "--quiet"]) == 0
    with pytest.raises(SystemExit) as excinfo:
        main([str(first), "-o", str(out), "--max-time", "0.1", "--quiet", "--w", "0.2"])
    assert excinfo.value.code == 2
    assert "only w = 1/3" in capsys.readouterr().err
    assert not out.exists()

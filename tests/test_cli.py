"""Tests for the command line interface."""

from pathlib import Path

import pytest

from pbh.cli import build_parser, main


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

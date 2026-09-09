"""Tests for snapshot writers."""

import io
from pathlib import Path

import numpy as np
import pytest

from pbh.base import FloatArray, Status
from pbh.ms import MS, MSEulerian, MSLagrangian
from pbh.output import GnuplotWriter, NpzWriter, Snapshot, open_writer

InitialData = tuple[FloatArray, FloatArray, FloatArray]

QUANTITIES = [
    "index",
    "r",
    "u",
    "m",
    "rho",
    "rfull",
    "ufull",
    "mfull",
    "rhofull",
    "horizon",
    "cs+",
    "cs-",
    "cs0",
    "xi",
    "Q",
    "ephi",
]


def make_snapshot(xi: float, n: int = 4) -> Snapshot:
    return {"index": np.arange(n), "r": np.linspace(1, 2, n), "xi": xi}


def test_gnuplot_writer_format() -> None:
    out = io.StringIO()
    GnuplotWriter(out).write(make_snapshot(0.5, n=2))
    assert out.getvalue() == "# index\tr\txi\n0\t1.0\t0.5\n1\t2.0\t0.5\n\n"


def test_npz_writer_stacks_snapshots(tmp_path: Path) -> None:
    path = tmp_path / "out.npz"
    with NpzWriter(path) as writer:
        writer.write(make_snapshot(0.0))
        # The first snapshot is saved straight away so the file exists early
        with np.load(path) as data:
            assert data["xi"] == pytest.approx([0.0])
        writer.write(make_snapshot(0.5))
        with np.load(path) as data:
            assert data["xi"] == pytest.approx([0.0])  # later snapshots wait for the context to close
    assert not path.with_name("out.npz.tmp").exists()
    with np.load(path) as data:
        assert set(data.files) == {"index", "r", "xi"}
        assert data["xi"] == pytest.approx([0.0, 0.5])
        assert data["r"].shape == (2, 4)
        assert data["index"].tolist() == [0, 1, 2, 3]


def test_npz_writer_with_no_snapshots(tmp_path: Path) -> None:
    path = tmp_path / "empty.npz"
    with NpzWriter(path):
        pass
    with np.load(path) as data:
        assert data.files == []


def test_open_writer_chooses_by_suffix(tmp_path: Path) -> None:
    with open_writer(tmp_path / "a.npz") as writer:
        assert isinstance(writer, NpzWriter)
    with open_writer(tmp_path / "a.dat") as writer:
        assert isinstance(writer, GnuplotWriter)


def test_snapshot_has_all_quantities(small_initial_data: InitialData) -> None:
    r, u, m = small_initial_data
    driver = MS(eomhandler=MSEulerian)
    driver.set_initial_conditions(0.0, r, u, m)
    snapshot = driver.snapshot()
    assert list(snapshot) == QUANTITIES
    assert snapshot["xi"] == 0.0
    assert np.asarray(snapshot["rho"]).shape == r.shape


def test_drive_writes_same_data_to_both_formats(tmp_path: Path, small_initial_data: InitialData) -> None:
    r, u, m = small_initial_data
    text = io.StringIO()
    npz = tmp_path / "out.npz"
    for writer in (GnuplotWriter(text), NpzWriter(npz)):
        driver = MS(eomhandler=MSLagrangian)
        driver.set_initial_conditions(0.0, r, u, m)
        driver.drive(output_step=0.5, writer=writer, max_time=1.0)
        if isinstance(writer, NpzWriter):
            writer.save()
    blocks = text.getvalue().strip().split("\n\n")
    with np.load(npz) as data:
        assert data["xi"] == pytest.approx([0.0, 0.5, 1.0])
        for k, block in enumerate(blocks):
            table = np.loadtxt(io.StringIO(block))
            for column, name in enumerate(QUANTITIES):
                expected = data[name] if name == "index" else data[name][k]
                assert table[:, column] == pytest.approx(expected)


@pytest.mark.parametrize("suffix", [".npz", ".dat"])
def test_load_initial_conditions_by_snapshot(tmp_path: Path, suffix: str, small_initial_data: InitialData) -> None:
    r, u, m = small_initial_data
    driver = MS(eomhandler=MSEulerian)
    driver.set_initial_conditions(0.0, r, u, m)
    path = tmp_path / f"out{suffix}"
    with open_writer(path) as writer:
        driver.drive(output_step=0.5, writer=writer, max_time=1.0)
    final = driver.snapshot()

    first = MS(eomhandler=MSEulerian)
    first.load_initial_conditions(path)
    assert first.status == Status.READY
    assert first.xi == 0.0
    assert first.eomhandler.r == pytest.approx(r)
    assert first.eomhandler.u == pytest.approx(u)
    assert first.eomhandler.m == pytest.approx(m)

    last = MS(eomhandler=MSEulerian)
    last.load_initial_conditions(path, snapshot=-1)
    assert last.xi == pytest.approx(1.0)
    assert last.eomhandler.m == pytest.approx(final["m"])

    middle = MS(eomhandler=MSEulerian)
    middle.load_initial_conditions(path, snapshot=1)
    assert middle.xi == pytest.approx(0.5)


def test_resumed_run_matches_uninterrupted_run(tmp_path: Path, small_initial_data: InitialData) -> None:
    r, u, m = small_initial_data
    full = MS(eomhandler=MSLagrangian)
    full.set_initial_conditions(0.0, r, u, m)
    full.drive(output_step=0.5, max_time=1.0)

    path = tmp_path / "half.npz"
    half = MS(eomhandler=MSLagrangian)
    half.set_initial_conditions(0.0, r, u, m)
    with NpzWriter(path) as writer:
        half.drive(output_step=0.5, writer=writer, max_time=0.5)
    resumed = MS(eomhandler=MSLagrangian)
    resumed.load_initial_conditions(path, snapshot=-1)
    resumed.drive(output_step=0.5, max_time=1.0)

    # The adaptive step history differs, so agreement is to integration tolerance rather than bit-exact
    assert resumed.xi == pytest.approx(full.xi)
    assert resumed.eomhandler.m == pytest.approx(full.eomhandler.m, rel=1e-6)

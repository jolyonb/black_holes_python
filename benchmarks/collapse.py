"""Wall-clock benchmark: a supercritical collapse from its initial data to the mass read-out, on both engines.

    uv run --group rust python benchmarks/collapse.py            # N = 200, 400, 800, 1600, best of 3
    uv run --group rust python benchmarks/collapse.py 400 1600 --repeats 5

The collapse is the Gaussian `A = 0.2`, `ell = 2` on `Rtilde_max = 30` with the production sinh stretch (scale 3), run
with `snapshots: milestones` until the mass is read. Each case is timed from `pbh run` to its end, the best of the
repeats, and the two engines' records (events, step and horizon tables) are compared: they must be identical. The
numbers depend on the machine and on its load; compare runs made on the same machine, idle.
"""

import argparse
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from pbh.cli import main
from pbh.driver import RunPaths
from pbh.output import RunReader
from pbh.timestep import Engine

CONFIG = (
    "grid: {{N: {N}, Rtilde_max: 30.0, scale: 3.0}}\noutput: {{snapshots: milestones}}\nevolution: {{xi_end: 9.0}}\n"
)


def same_records(a: RunReader, b: RunReader) -> bool:
    """Whether two runs recorded the same events, step table and horizon table."""
    if a.events != b.events:
        return False
    for table in ("steps", "horizon"):
        ta, tb = getattr(a, table), getattr(b, table)
        for name, column in ta.items():
            if isinstance(column, list):
                if column != tb[name]:
                    return False
            elif not np.array_equal(column, np.asarray(tb[name]), equal_nan=True):
                return False
    return True


def run(directory: Path, N: int, engine: Engine) -> float:
    """One timed run of the collapse at `N` on `engine`, from its initial file to the end; the wall time."""
    name = f"n{N}_{engine.value}"
    config = directory / f"n{N}.yaml"
    config.write_text(CONFIG.format(N=N))
    args = ["initial", "gaussian", name, "--config", str(config), "--A", "0.2", "--ell", "2.0", "--dir", str(directory)]
    assert main(args) == 0
    start = time.perf_counter()
    assert main(["run", str(config), name, "--engine", engine.value, "--dir", str(directory)]) == 0
    return time.perf_counter() - start


def machine() -> str:
    """The platform, the processor (its brand name where the system says it) and the Python version."""
    chip = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
        chip = brand.strip() or chip
    return f"{platform.platform()}, {chip}, Python {platform.python_version()}"


def main_benchmark() -> None:
    """Time every case and print the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("N", type=int, nargs="*", default=[200, 400, 800, 1600])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(machine())
    print(f"commit {commit}, best of {args.repeats}\n")
    print("|    N | steps | numpy | Rust | numpy per step | Rust per step | Rust faster | records |")
    print("|---:|---:|---:|---:|---:|---:|---:|---|")
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for N in args.N:
            best = {engine: min(run(directory, N, engine) for _ in range(args.repeats)) for engine in Engine}
            readers = {engine: RunReader(RunPaths.of(directory, f"n{N}_{engine.value}").evolution) for engine in Engine}
            steps = len(readers[Engine.PYTHON].steps["step"])
            same = same_records(readers[Engine.PYTHON], readers[Engine.RUST])
            numpy_s, rust_s = best[Engine.PYTHON], best[Engine.RUST]
            print(
                f"| {N} | {steps} | {numpy_s:.2f} s | {rust_s:.2f} s | {1e6 * numpy_s / steps:.0f} us "
                f"| {1e6 * rust_s / steps:.0f} us | {numpy_s / rust_s:.1f}x | {'identical' if same else 'DIFFER'} |",
                flush=True,
            )


if __name__ == "__main__":
    main_benchmark()

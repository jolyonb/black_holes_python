"""The command line: `pbh validate`, `pbh initial gaussian`, `pbh run` and `pbh restart`.

Every command works on a run directory and a run name, and a run is its three files, `name.config.yaml`,
`name.initial.h5` and `name.evolution.h5`:

    pbh validate CONFIG                              parse a configuration and print it with every default filled in
    pbh initial gaussian NAME --config CONFIG --A A --ell ELL [--xi0 XI]
                                                     write NAME.initial.h5: the growing mode of a Gaussian delta_m
    pbh run CONFIG NAME                              run CONFIG from NAME.initial.h5, writing the other two files
    pbh restart SOURCE NAME [--snapshot I] [--config CONFIG]
                                                     start the run NAME from a snapshot of the run SOURCE (the last
                                                     by default), with SOURCE's configuration unless another is given

The Gaussian command is a convenience for the paper's standard perturbation; any other datum is written with
`records.write_initial` from Python, since the initial data are the initial data however they were made. No flag
overrides a configuration value: the configuration file is the record of the run, and editing it is the honest
way to change one.

`main` is the entry point twice over: `pyproject.toml` registers it as the `pbh` console script, and running the
module directly, `python -m pbh.cli`, reaches it through the block at the bottom.
"""

import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import yaml
from cyclopts import App, CycloptsError, Parameter

from pbh.config import ConfigError, RunConfig, load
from pbh.driver import RunPaths, RunResult
from pbh.driver import run as run_driver
from pbh.eos import Background
from pbh.initial import GrowingMode, IllPosedDataError, NotCompensatedError
from pbh.output import RunReader
from pbh.profiles import Gaussian
from pbh.records import StateRecord, read_initial, write_initial

app = App(name="pbh", help="Primordial black hole formation: Misner-Sharp evolution of a perturbed FRW fluid.")
initial = App(name="initial", help="Write an initial-data file.")
app.command(initial)

type Directory = Annotated[Path, Parameter(name="--dir", help="The run directory, where a run's three files live.")]


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point; returns the exit status: 0 done, 1 a refused input, 2 an aborted run or a usage error."""
    try:
        command, bound, _ = app.parse_args(argv, exit_on_error=False)
        command(*bound.args, **bound.kwargs)
    except SystemExit as leaving:  # --help, or an aborted run
        return int(leaving.code or 0)
    except (ConfigError, IllPosedDataError, NotCompensatedError, ValueError, FileNotFoundError) as refused:
        print(f"pbh: {refused}", file=sys.stderr)
        return 1
    except CycloptsError:  # cyclopts has already printed the usage error
        return 2
    return 0


@app.command
def validate(config: Annotated[Path, Parameter(help="The configuration file.")]) -> None:
    """Parse a configuration and print it with every default filled in."""
    parsed = load(config)
    print(yaml.safe_dump(parsed.model_dump(mode="json", exclude_none=True), sort_keys=False), end="")


@initial.command
def gaussian(
    name: Annotated[str, Parameter(help="The run whose initial file to write.")],
    *,
    config: Annotated[Path, Parameter(help="The configuration whose grid and fluid the data are made for.")],
    A: Annotated[float, Parameter(name="--A", help="The amplitude of delta_m = A exp(-X^2 / 2 ell^2).")],
    ell: Annotated[float, Parameter(help="The width.")],
    xi0: Annotated[float, Parameter(help="The start time; the run begins there.")] = 0.0,
    dir: Directory = Path(),
) -> None:
    """Write the growing mode of a Gaussian mass profile, the paper's standard perturbation (Section 7.9)."""
    parsed = load(config)
    paths = RunPaths.of(dir, name)
    sch = parsed.scheme()
    if not sch.eos.is_radiation:
        raise ValueError("the growing-mode data of Section 7.9 exist for radiation only")
    bg_0 = Background.at(sch.eos, xi0)
    mode, report = GrowingMode.from_profile(Gaussian(A=A, ell=ell), "m", parsed.grid.Rtilde_max, bg_0)
    geo = sch.frame(xi0).geo
    state = mode.state(geo, bg_0)
    ratio = mode.correction_ratio(geo, bg_0)
    provenance = {
        "method": "growing_mode",
        "field": "m",
        "profile": "gaussian",
        "A": A,
        "ell": ell,
        "xi_0": xi0,
        "nonlinear_correction": True,
        "power_fraction": report.power_fraction,
        "amplified_fraction": report.amplified_fraction,
        "distance_to_first_zero": report.distance_to_first_zero,
        "edge_value": report.edge_value,
        "correction_ratio": ratio,
    }
    write_initial(paths.initial, StateRecord.of(state, geo.X[: geo.N + 1], xi0, 0, provenance))
    peak = 2.0 * ell**2 * A / math.e  # Section 5.4: the Gaussian's peak compaction at xi = 0
    print(f"wrote {paths.initial}: peak compaction {peak:.4f}, correction ratio {ratio:.2%}")


@app.command
def run(
    config: Annotated[Path, Parameter(help="The configuration file.")],
    name: Annotated[str, Parameter(help="The run: NAME.initial.h5 is read, the other two files are written.")],
    *,
    dir: Directory = Path(),
) -> None:
    """Run a configuration from the run's initial-data file to its end."""
    paths = RunPaths.of(dir, name)
    report(run_driver(load(config), read_initial(paths.initial), paths))


@app.command
def restart(
    source: Annotated[str, Parameter(help="The run to start from.")],
    name: Annotated[str, Parameter(help="The new run.")],
    *,
    snapshot: Annotated[int, Parameter(help="Which snapshot of SOURCE; negative counts from the end.")] = -1,
    config: Annotated[Path | None, Parameter(help="A configuration to use instead of SOURCE's.")] = None,
    dir: Directory = Path(),
) -> None:
    """Start a new run from a snapshot of another, with its configuration unless another is given."""
    reader = RunReader(RunPaths.of(dir, source).evolution)
    parsed: RunConfig = load(config) if config is not None else reader.config
    record = reader.snapshot(snapshot % len(reader.snapshots))
    paths = RunPaths.of(dir, name)
    write_initial(paths.initial, record)
    report(run_driver(parsed, read_initial(paths.initial), paths))


def report(result: RunResult) -> None:
    """Print how a run ended; an abort leaves with the exit status 2."""
    print(f"{result.status}: {result.steps} steps to xi = {result.xi:.6f}; wrote {result.paths.evolution}")
    if result.status != "completed":
        raise SystemExit(2)


if __name__ == "__main__":  # `python -m pbh.cli ...`; the `pbh` script of pyproject.toml calls `main` the same way
    sys.exit(main())

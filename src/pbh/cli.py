"""The command line: `pbh validate`, `pbh initial gaussian`, `pbh run`, `pbh restart` and `pbh summary`.

Every command works on a run directory and a run name, and a run is its three files, `name.config.yaml`,
`name.initial.h5` and `name.evolution.h5`:

    pbh validate CONFIG                              parse a configuration and print it with every default filled in
    pbh initial gaussian NAME --config CONFIG --A A --ell ELL [--pair]
                                                     write NAME.initial.h5: the growing mode of a Gaussian delta_m,
                                                     given at the configuration's xi_start; with --pair also
                                                     NAME.half.initial.h5, the same datum on the grid with N halved
    pbh run CONFIG NAME [--engine E] [--pair]        run CONFIG from NAME.initial.h5, writing the other two files;
                                                     with --pair then run NAME.half at N/2 from NAME.half.initial.h5
    pbh restart SOURCE NAME [--snapshot I] [--config CONFIG] [--engine E]
                                                     start the run NAME from a snapshot of the run SOURCE (the last
                                                     by default), with SOURCE's configuration unless another is given
    pbh summary NAME [--export JSON]                 what NAME says about its black hole, recomputed from its
                                                     evolution file and printed, and written as JSON on request;
                                                     with the pair's analysis if NAME.half.evolution.h5 exists and
                                                     is NAME's companion

The Gaussian command is a convenience for the paper's standard perturbation; any other datum is written with
`records.write_initial` from Python, since the initial data are the initial data however they were made. No flag
overrides a configuration value: the configuration file is the record of the run, and editing it is the honest
way to change one. `--pair` is not one either: it runs a second run, the half companion `NAME.half`, whose
configuration is derived from the given one by halving `grid.N` (`pair.half_config`) and saved as its own file.
`--engine` is not one: it chooses which implementation evaluates the stages, numpy (`python`, the
default) or the optional Rust engine (`rust`), which compute the same numbers, and both files record it.

`main` is the entry point twice over: `pyproject.toml` registers it as the `pbh` console script, and running the
module directly, `python -m pbh.cli`, reaches it through the block at the bottom.
"""

import json
import math
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, cast

import yaml
from cyclopts import App, CycloptsError, Parameter

from pbh.config import ConfigError, RunConfig, load
from pbh.driver import RunPaths, RunResult, core_watch, epoch_history
from pbh.driver import run as run_driver
from pbh.eos import Background
from pbh.initial import GrowingMode, IllPosedDataError, NotCompensatedError
from pbh.output import RunReader
from pbh.pair import analyse, half_config, half_name
from pbh.pair import as_json as pair_json
from pbh.pair import describe as describe_pair
from pbh.profiles import Gaussian
from pbh.records import StateRecord, read_initial, write_initial
from pbh.summary import as_json, describe, summarise
from pbh.timestep import Engine

app = App(name="pbh", help="Primordial black hole formation: Misner-Sharp evolution of a perturbed FRW fluid.")
initial = App(name="initial", help="Write an initial-data file.")
app.command(initial)

type Directory = Annotated[Path, Parameter(name="--dir", help="The run directory, where a run's three files live.")]
type EngineChoice = Annotated[
    Engine, Parameter(help="python (numpy, the reference) or rust (the optional compiled engine; the same numbers).")
]
type PairChoice = Annotated[
    bool, Parameter(negative=(), help="Also the half companion NAME.half at grid.N / 2, for the pair's error estimate.")
]


def main(argv: Sequence[str] | None = None) -> int:
    """The entry point; returns the exit status: 0 done, 1 a refused input, 2 an aborted run or a usage error."""
    try:
        command, bound, _ = app.parse_args(argv, exit_on_error=False)
        command(*bound.args, **bound.kwargs)
    except SystemExit as leaving:  # --help, or an aborted run
        return int(leaving.code or 0)
    except (
        ConfigError,
        IllPosedDataError,
        NotCompensatedError,
        ValueError,
        FileNotFoundError,
        ModuleNotFoundError,
    ) as refused:
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
    pair: PairChoice = False,
    dir: Directory = Path(),
) -> None:
    """Write the growing mode of a Gaussian mass profile, the paper's standard perturbation (Section 7.9).

    The profile is given at the configuration's `evolution.xi_start`, where the run begins. With `--pair` the same
    datum is also written on the grid of the configuration with `grid.N` halved, for the companion `NAME.half`.
    """
    parsed = load(config)
    made = [(RunPaths.of(dir, name), *gaussian_record(parsed, A, ell, {}))]
    if pair:  # both data are made before either is written, so a refused companion leaves nothing behind
        half = gaussian_record(half_config(parsed), A, ell, {"half_of": name})
        made.append((RunPaths.of(dir, half_name(name)), *half))
    for paths, record, peak in made:
        write_initial(paths.initial, record)
        ratio = record.provenance["correction_ratio"]
        print(f"wrote {paths.initial}: peak compaction {peak:.4f}, correction ratio {ratio:.2%}")


def gaussian_record(parsed: RunConfig, A: float, ell: float, extra: dict[str, str]) -> tuple[StateRecord, float]:
    """The Gaussian's growing mode on the configuration's grid, `extra` in its provenance, and its peak compaction."""
    xi0 = parsed.evolution.xi_start
    sch = parsed.scheme()
    if not sch.eos.is_radiation:
        raise ValueError("the growing-mode data of Section 7.9 exist for radiation only")
    bg_0 = Background.at(sch.eos, xi0)
    mode, report = GrowingMode.from_profile(Gaussian(A=A, ell=ell), "m", parsed.grid.Rtilde_max, bg_0)
    geo = sch.frame(xi0).geo
    deviation = mode.deviation(geo, bg_0)  # never the state: a perturbation below round-off of FRW must survive
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
        **extra,
    }
    peak = 2.0 * ell**2 * A / (math.e * math.exp(xi0))  # Section 5.4: X^2 delta_m / Rtilde_H^2 at sqrt 2 ell, at xi0
    return StateRecord.of_deviation(deviation, geo.X[: geo.N + 1], xi0, provenance), peak


@app.command
def run(
    config: Annotated[Path, Parameter(help="The configuration file.")],
    name: Annotated[str, Parameter(help="The run: NAME.initial.h5 is read, the other two files are written.")],
    *,
    engine: EngineChoice = Engine.PYTHON,
    pair: PairChoice = False,
    dir: Directory = Path(),
) -> None:
    """Run a configuration from the run's initial-data file to its end; with `--pair` then its half companion.

    Both initial files are read before either run starts, and the companion's must name NAME as the run it is the
    half of; the companion then runs whether the first run completes or aborts (an exception ends both).
    """
    parsed = load(config)
    paths = RunPaths.of(dir, name)
    initial = read_initial(paths.initial)
    if not pair:
        report(run_driver(parsed, initial, paths, engine=engine))
        return
    half, half_paths = half_config(parsed), RunPaths.of(dir, half_name(name))
    half_initial = read_initial(half_paths.initial)  # read before the first run, so a missing file costs nothing
    if half_initial.provenance.get("half_of") != name:
        raise ValueError(
            f"{half_paths.initial} is not the half datum of {name}: its provenance does not say half_of {name}"
        )
    first = run_driver(parsed, initial, paths, engine=engine)
    report(first, run_driver(half, half_initial, half_paths, engine=engine, half_of=name))


@app.command
def restart(
    source: Annotated[str, Parameter(help="The run to start from.")],
    name: Annotated[str, Parameter(help="The new run.")],
    *,
    snapshot: Annotated[int, Parameter(help="Which snapshot of SOURCE; negative counts from the end.")] = -1,
    config: Annotated[Path | None, Parameter(help="A configuration to use instead of SOURCE's.")] = None,
    engine: EngineChoice = Engine.PYTHON,
    dir: Directory = Path(),
) -> None:
    """Start a new run from a snapshot of another, with its configuration unless another is given.

    The engine is not inherited from SOURCE: it is this command's `--engine`, since the engines agree.
    """
    reader = RunReader(RunPaths.of(dir, source).evolution)
    parsed: RunConfig = load(config) if config is not None else reader.config
    record = reader.snapshot(snapshot % len(reader.snapshots))
    paths = RunPaths.of(dir, name)
    write_initial(paths.initial, record)
    history = epoch_history(reader, record.xi)  # the M_AH series the read-out needs, from before the snapshot
    core = core_watch(reader, record.xi)  # and the core's series, for the bounce
    report(run_driver(parsed, read_initial(paths.initial), paths, history, core, engine))


@app.command
def summary(
    name: Annotated[str, Parameter(help="The run whose evolution file is read.")],
    *,
    export: Annotated[Path | None, Parameter(help="Also write the summary and its series as JSON here.")] = None,
    dir: Directory = Path(),
) -> None:
    """What the run says about its black hole, recomputed from its evolution file (`summary.py`); nothing is stored.

    If the half companion `NAME.half` has an evolution file, the pair's analysis follows (`pair.py`) once both runs
    have ended; until then the run is summarised alone and the pair said to be waiting. A file left over from another
    run, at another `N` or configuration, or from before NAME was last run, is said not to be the companion
    (`is_companion`) and is not analysed.
    """
    reader = RunReader(RunPaths.of(dir, name).evolution)
    summary = summarise(reader)
    data = as_json(summary)
    lines = [describe(summary)]
    companion = RunPaths.of(dir, half_name(name))
    if companion.evolution.exists():
        half_reader = RunReader(companion.evolution)
        half = summarise(half_reader)
        if not is_companion(name, reader, half_reader, companion.config):
            lines.append(f"pair: {half_name(name)} is not this run's companion; no pair analysis")
        elif summary.status is None or half.status is None:
            waiting = name if summary.status is None else half_name(name)
            lines.append(f"pair: {waiting} has not ended; no pair analysis yet")
        else:
            analysed = analyse(summary, half, reader.config)
            lines += describe_pair(analysed)
            data["pair"] = pair_json(analysed)
    print("\n".join(lines))
    if export is not None:
        export.write_text(json.dumps(data, indent=1))


def is_companion(name: str, reader: RunReader, half: RunReader, saved: Path) -> bool:
    """Whether the run read by `half`, its configuration saved at `saved`, is the companion of the run `name`.

    It is if its provenance says it is the half of `name`, its configuration is `name`'s with `N` halved, and it was
    started after `name` was, as `pbh run --pair` starts it; a rerun of or a restart into `name` makes it stale.
    """
    if not saved.exists():
        return False
    provenance = cast(dict[str, dict[str, Any] | None], yaml.safe_load(saved.read_text())).get("provenance") or {}
    halved = half.config.grid.N * 2 == reader.config.grid.N and half.config == half_config(reader.config)
    started_after = datetime.fromisoformat(half.written) > datetime.fromisoformat(reader.written)
    return provenance.get("half_of") == name and halved and started_after


def report(*results: RunResult) -> None:
    """Print how each run ended; an abort of any leaves with the exit status 2."""
    for result in results:
        print(f"{result.status}: {result.steps} steps to xi = {result.xi:.6f}; wrote {result.paths.evolution}")
    if any(result.status != "completed" for result in results):
        raise SystemExit(2)


if __name__ == "__main__":  # `python -m pbh.cli ...`; the `pbh` script of pyproject.toml calls `main` the same way
    sys.exit(main())

# pbh: primordial black hole formation

Evolution of a spherically symmetric perturbation of a flat FRW perfect fluid, `P = w rho`, through the formation of
a black hole and its subsequent accretion, in the Misner-Sharp (fluid-orthogonal) slicing. After a horizon forms the
code excises the trapped region and follows the hole on a grid pinned to physical radius near it, and reads the
hole's mass as the rate-corrected estimate `M_est = M_AH / (1 - d ln M_AH / d xi)`, with an error bar formed from the
same record.

The numerical scheme is specified in the companion paper (Sections 7 and 8, kept outside this repository): every
module's docstring names the section and the equations it implements, and the paper is the specification the code
answers to. Units are `R_H = 1`, the Hubble radius at `xi = 0`; `xi = ln(t / t_0)`; tilde variables are scaled by
the background (`rhotilde = rho / rho_FRW`, `Rtilde = R / (a R_H)`, `Utilde = U / (H a R_H)`), so FRW is
`rhotilde = 1`, `Utilde = Rtilde`.

## Getting started

The project is managed with [uv](https://docs.astral.sh/uv/) (0.8.6 or newer) and requires Python 3.14. `pbh` is
pure Python and runs on its numpy engine alone. The Rust engine (`pbh run ... --engine rust`, the same stage compiled,
about twice as fast) is a separate, optional package, `pbh-engine` in `rust/`, installed by the dependency group
`rust`; building it needs a Rust toolchain, 1.85 or newer, with clippy and rustfmt (`rustup`).

```
uv sync                     # pure Python: no Rust toolchain needed
uv sync --group rust        # with the Rust engine (needs cargo); pass --group rust to `uv run` as well
uv run pbh --help
```

`uv sync` without `--group rust` removes the engine again, and `uv run` without it neither installs nor rebuilds it,
so for work with the engine pass the flag to both (`uv run --group rust pytest`). Without the engine everything runs
on numpy: the Rust engine's tests are skipped, and `--engine rust` is refused with the
command that installs it. Where there is no `cargo`, `--group rust` fails at once rather than download a toolchain
(about 500 MB); that guard is `[tool.uv.extra-build-variables]` in `pyproject.toml`, which a uv older than 0.8.6
ignores.

A run is three files in one directory, named after the run: `NAME.config.yaml`, the configuration as the run saw it;
`NAME.initial.h5`, the initial data; and `NAME.evolution.h5`, written as the run goes and readable while it does. A
supercritical collapse from start to mass, in about ten seconds:

```
cat > collapse.yaml <<EOF
grid: {N: 200, Rtilde_max: 30.0, scale: 3.0}
evolution: {xi_end: 9.0}
EOF
uv run pbh validate collapse.yaml                                           # the configuration with every default
uv run pbh initial gaussian collapse --config collapse.yaml --C 0.5886 --ell 2   # a Gaussian seed, its start
uv run pbh run collapse.yaml collapse                                        # formation, excision, the mass
uv run pbh summary collapse                                                  # what the run says about its hole
```

The data start at `xi = -9.43` (below). The run stops once the mass is read, here at `xi = 7.3`, 3.5 e-folds after
formation, with `M_est = 11.18 R_H` and an error bar under one per cent. `pbh summary` recomputes everything from the evolution file (nothing derived is stored):
the core before any horizon (formed, bounced or undecided; the peak of the physical central density and the resolution
there), the quoted reading, the long-run reference, when the bar crossed 5, 1 and 0.3 per cent, a fit of the accretion
law, the enclosed-mass cross-check on spheres of fixed physical radius, the fold monitor's nearest approach after
excision, and the outer boundary's reach: the `Rtilde_max` that keeps the apparent horizon (at formation and at the
reading) and the origin (at the end) out of the boundary's sound and light cones, and whether the run's clears it.
`--export FILE.json` writes it with its series.

The initial data are the growing mode of a time-independent seed, the first-order profile `delta_m0(X)` of the
growing solution `delta_m = e^xi delta_m0 + O(eps^4)` (radiation; `initial.py`), here a Gaussian in `delta_m0`, which
is a Gaussian in the curvature profile `K` of the literature (`delta_m0 = 2 K / 3`), its compaction `X^2 delta_m0`
peaking at `r_m = sqrt 2 ell` with the value `C = 2 ell^2 A / e`. The seed fixes the start: the data begin at the latest
time with `eps0^2 = (R_H / r_m)^2` at or below `initial.epsilon2` (default `1e-5`, `xi = ln 8e-5 = -9.43` here), each
mode grown exactly from the seed and the second-order terms added, so they are the growing solution to relative
`O(eps0^4)`. The run writes that start into its saved configuration as `evolution.xi_start`, which a configuration
normally leaves out (but must give to `pbh run` a snapshot of another run as its initial file; `pbh restart` does it
for you). However small the perturbation at the start (`initial: {epsilon2: 1.0e-14}` puts a collapse-sized
Gaussian at `delta_m ~ 1e-14`), the data are built and recorded as their deviation from FRW, so nothing is lost, and the
early steps are few, since there the step cap and not the sound speed sets them.

`pbh restart SOURCE NAME [--snapshot K] [--config OTHER.yaml]` starts a new run from any snapshot of another, with its
configuration or a different one: every snapshot is a restart point, and a restart carries the read-out's history.

A run's spatial error comes from a pair: `pbh initial gaussian NAME ... --pair` also writes `NAME.half.initial.h5`, the
same datum on the grid with `grid.N` halved (`N` must be even), and `pbh run CONFIG NAME --pair` then runs the companion
`NAME.half` at `N/2`, about a quarter more work, saving its derived configuration with `half_of: NAME` in its
provenance. `pbh summary NAME` then adds the pair's analysis (`pair.py`, Section 8.6), provided `NAME.half` is this
run's companion (its provenance names `NAME`, its configuration is `NAME`'s with `N` halved, and it started after
`NAME`; a leftover from an earlier run is said not to be): both outcomes and whether they agree (an abort never
does); the mass at `N` and `N/2` with its calibrated error `F (v_N - v_(N/2))/3`, the peak central density with the
bare third (uncalibrated), the formation and bounce times without one (they are sampled at the steps); whether the
run is far enough from threshold to trust its side of it, with the probability and the `N` that would reach 95 per
cent, and a caveat wherever the step cap is looser than the calibration's `1e-7` or the start tolerance
`initial.epsilon2` looser than `1e-5` (the seed's start costs nothing below it); and `M_est` with
its error budget, spatial, read-out bar and quoted systematics in quadrature, with what the total does not cover. The
trust width, the coverage factor and the systematics are calibrated by the campaigns of bite 26; `pair.py` names the
source of each.

`pbh run` and `pbh restart` take `--engine python|rust`: which implementation evaluates the stages, numpy (the
reference and the default) or the optional Rust engine (the same numbers, about twice as fast on a whole run). It is
not physics and not part of the configuration, so a configuration runs on any machine; each run records its engine in
the `provenance` of `NAME.config.yaml` and in the evolution file, and a restart uses its own `--engine`, not its
source's.

## Configuration

`examples/example.config.yaml` lists every key with its default. A configuration must state `grid.N`,
`grid.Rtilde_max` and `evolution.xi_end`; everything else has the paper's production value. The sections:

* `fluid`: `w` as an exact rational (`1/3` is radiation). The exact outgoing-wave outer condition exists for
  radiation only and is refused otherwise.
* `grid`: the static map, `sinh` (fine at the origin, coarse in the background; the production choice) or
  `uniform`; `N` cells out to `Rtilde_max`.
* `outer`: the outer boundary, the outgoing-wave condition as a penalty or the state held at FRW.
* `shocks`: the shock-capturing kernels and their constants (the theta-limiter's `theta`, the limiter, `c_v`), or
  `centred` for the base scheme.
* `excision`: the switch-on and the pinned map (`eta`, `tau_on`, `c_t`, `c_Delta`), re-excision, and `enabled:
  false` to continue a collapse unexcised, for comparison with the excised run.
* `readout`: the rate window, the two-e-fold floor, the error bar and the target at which the mass is read, and
  whether the run stops there.
* `stepping`: RK4's Courant number and the step cap.
* `output`: the snapshot schedule (uniform in `xi` before formation, in physical time after), the flush cadence, and
  whether the full monitor record is written every step or only at snapshots.
* `evolution`: `xi_end`, and `stop_on_bounce` to end a sub-threshold run once its core has bounced (the central
  density halved from its peak and held there for 0.5 in `xi`), as a threshold study wants.

Floats need a digit on both sides of the point and after an exponent sign (`1.0e-5`, not `1e-5`); unknown keys are
errors, reported with the file and every bad key.

## The evolution file

HDF5, in single-writer multiple-reader mode, with four tables that share nothing but the step number:

| table | rows | what |
|---|---|---|
| `steps` | one per step | `xi`, `dxi`, what limited the step, refused attempts, and the monitors (conservation, the outer boundary, stability, positivity, resolution) |
| `events` | one per event | a kind and a JSON payload: `formation`, `switch_on`, `re_excision`, `readout`, `bounce`, `rejection`, `abort`, `end`, ... |
| `snapshots` | one per output time | the integrator's variables only (deviations from FRW, and the content itself of the cells stored whole in `E_whole`), from which every derived field is recomputed |
| `horizon` | one per step | the finder's report: `M_AH`, `X_AH`, the trapping margins, the excision face, the fold monitor, the near-zone monitors |

```python
from pathlib import Path

from pbh.output import RunReader

run = RunReader(Path("collapse.evolution.h5"))
xi, M_AH = run.horizon["xi"], run.horizon["M_AH"]
readout = [e.payload for e in run.events if e.kind == "readout"]
state = run.snapshot(-1)  # a StateRecord: restartable, and the input to any derived field
config = run.config  # the configuration the run used, to rebuild its Scheme
```

What each column means, and which paper equation it comes from, is in the docstrings of `pbh.monitors`,
`pbh.horizon` and `pbh.output`; what the code must log for every number in the paper to be regenerated is specified
alongside the paper (`PRODUCTION_OUTPUT_SPEC.md`).

A run ends in one of three ways, each recorded as the `end` event: completed (at `xi_end`, when the mass was read,
or when the core bounced); aborted, with a named cause (no step passes the checks after twenty halvings; the clock stalls, a step lost even in
the compensated sum of `xi`; a chart failure, named by case; a switch-on transition that cannot fit, naming the
radius it needs; an excision face that is no longer an outflow boundary, or its three faces no longer trapped; the fold
monitor, a shock behind which the slice folds, naming the cure, to excise further out); or interrupted, by an exception (Ctrl-C included), after a final flush.

## The code

```
src/pbh/
  maps, geometry, layout, state      the grid map X(xi, x), exact cell geometry, the packed state, FRW
  storage                            each cell stored as its deviation, or whole once far below the background
  stencils, kernels, derived         difference quotients in X and s = X^2, the shock-capturing kernels, derived fields
  equations                          one stage: the semi-discrete equations, in deviation form
  outer                              the outer closure (outgoing-wave penalty, or held at FRW)
  timestep                           RK4 with every stage checked, the Courant step and the step cap
  horizon, excision                  the finder, the switch-on and re-excision, the pinned map's zones
  collapse                           the core before formation: the peak central density and the bounce
  driver, cli                        the run loop and the command line
  config, initial, profiles          the configuration; initial data (the growing mode of a seed)
  records, output, h5                initial and snapshot records; the evolution file
  monitors, readout, summary         per-step monitors; the mass read-out; the run summary
  pair                               the pair at N and N/2: spatial errors, threshold trust, the mass error budget
  causal                             how far the outer boundary can have reached, on sound and on light
  michel                             the Michel accretion flow, the late-time background and a test
  rust_engine                        the adapter to the Rust engine
rust/                                the optional Rust engine, package pbh-engine (module pbh_engine, stubs pbh_engine.pyi)
  src/                               the stage of equations.py, one function per Python function
src/_old/                            the retired collocated code (2015-2026), kept for reference; not run
```

The evolved variables are the cell energy contents and the face velocities on a staggered grid (density on cells,
velocity and mass on faces), with the mass by cumulative sum, so the mass constraint holds by construction; the
integrator advances their deviation from FRW, which keeps the far field FRW to round-off, except in a cell below a
quarter of the background, which it stores whole (back as its deviation above a half), so that a void keeps its
precision however empty it gets.

### Development

```
uv run pytest               # the fast suite (about 15 s)
uv run pytest -m slow       # evolutions (about a minute)
uv run pytest -m ''         # everything
uv run ruff check . && uv run ruff format . && uv run pyright   # lint, format, strict types
uv run pre-commit install   # all of the above on every commit, and cargo fmt and clippy on the Rust
```

The gates (pyright, and the pre-commit hook's 100 per cent coverage, which includes `pbh.rust_engine`) run with the
Rust engine installed: the hooks pass `--group rust`. The engine is rebuilt by any `uv sync --group rust` or
`uv run --group rust` after a change to `rust/` (never
`maturin develop`; `cache-keys` in `rust/pyproject.toml`); its gates are `cargo fmt --check` and
`cargo clippy -- -D warnings` in `rust/`, and it is tested from pytest against the numpy engine
(`tests/test_rust_engine.py`). Run `uv sync --group rust` once before the cargo gates in a fresh clone: they build PyO3 against the
project's own interpreter, the one in `.venv` (`.cargo/config.toml`), which `uv sync` creates. The engine is built in
the project environment, with the `rust` group's maturin, not in an isolated one (`no-build-isolation-package` in
`pyproject.toml`), so that an edit recompiles only the engine. CI runs the gates with the engine, and the test suite
once more without it, on a pure-Python install; it pins the Rust toolchain (`.github/workflows/ci.yml`) and also
checks the crate on its declared minimum, 1.85; until that job has run, the minimum is declared, not tested (see
`rust/Cargo.toml`). `rust/target/` is the build tree: ignored, and refused by a pre-commit hook if it is ever staged.

Style: flat pytest functions, pyright strict, explicit ABCs, and no computer algebra in the code or its tests (exact
`Fraction` arithmetic where exactness is needed); the symbolic checks of the paper live with the paper.

## History

The original collocated code (2015; collocated centred derivatives, adaptive Dormand-Prince) lives on as `src/_old`. Its long-standing "high-frequency instability" was diagnosed in September 2026 as the odd-even null
mode of collocated centred first derivatives; the staggered layout of this code has no such mode, since the
sawtooth is its stiffest direction rather than an invisible one. The rebuild began on 2026-09-18.

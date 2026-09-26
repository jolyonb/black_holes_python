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

The project is managed with [uv](https://docs.astral.sh/uv/) and requires Python 3.14.

```
uv sync
uv run pbh --help
```

A run is three files in one directory, named after the run: `NAME.config.yaml`, the configuration as the run saw it;
`NAME.initial.h5`, the initial data; and `NAME.evolution.h5`, written as the run goes and readable while it does. A
supercritical collapse from start to mass, in about ten seconds:

```
cat > collapse.yaml <<EOF
grid: {N: 200, Rtilde_max: 12.0, scale: 3.0}
evolution: {xi_end: 8.0}
EOF
uv run pbh validate collapse.yaml                                         # the configuration with every default
uv run pbh initial gaussian collapse --config collapse.yaml --A 0.2 --ell 2   # the growing mode of a Gaussian
uv run pbh run collapse.yaml collapse                                      # formation, excision, the mass
uv run pbh summary collapse                                                # what the run says about its hole
```

The run stops once the mass is read, here at `xi = 7.3`, 3.4 e-folds after formation, with `M_est = 11.14 R_H` and
an error bar under one per cent. `pbh summary` recomputes everything from the evolution file (nothing derived is
stored): the core before any horizon (formed, bounced or undecided; the peak of the physical central density and the
resolution there), the quoted reading, the long-run reference, when the bar crossed 5, 1 and 0.3 per cent, a fit of the
accretion law, and the enclosed-mass cross-check on spheres of fixed physical radius. `--export FILE.json` writes it
with its series.

`pbh restart SOURCE NAME [--snapshot K] [--config OTHER.yaml]` starts a new run from any snapshot of another, with its
configuration or a different one: every snapshot is a restart point, and a restart carries the read-out's history.

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
| `snapshots` | one per output time | the integrator's variables only (deviations from FRW), from which every derived field is recomputed |
| `horizon` | one per step | the finder's report: `M_AH`, `X_AH`, the trapping margins, the excision face, the near-zone monitors |

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
or when the core bounced); aborted, with a named cause (a cell below `5e-13` of the background, where the fluid-orthogonal slicing and
the arithmetic both end; a chart failure, named by case; a switch-on transition that cannot fit, naming the
radius it needs); or interrupted, by an exception (Ctrl-C included), after a final flush.

## The code

```
src/pbh/
  maps, geometry, layout, state      the grid map X(xi, x), exact cell geometry, the packed state, FRW
  stencils, kernels, derived         difference quotients in X and s = X^2, the shock-capturing kernels, derived fields
  equations                          one stage: the semi-discrete equations, in deviation form
  outer                              the outer closure (outgoing-wave penalty, or held at FRW)
  timestep                           RK4 with every stage checked, the Courant step and the step cap
  horizon, excision                  the finder, the switch-on and re-excision, the pinned map's zones
  collapse                           the core before formation: the peak central density and the bounce
  driver, cli                        the run loop and the command line
  config, initial, profiles          the configuration; initial data (the growing mode of a mass profile)
  records, output, h5                initial and snapshot records; the evolution file
  monitors, readout, summary         per-step monitors; the mass read-out; the run summary
  michel                             the Michel accretion flow, the late-time background and a test
src/_old/                            the retired collocated code (2015-2026), kept for reference; not run
```

The evolved variables are the cell energy contents and the face velocities on a staggered grid (density on cells,
velocity and mass on faces), with the mass by cumulative sum, so the mass constraint holds by construction; the
integrator advances their deviation from FRW, which keeps the far field FRW to round-off.

### Development

```
uv run pytest               # the fast suite (about 15 s)
uv run pytest -m slow       # evolutions (about a minute)
uv run pytest -m ''         # everything
uv run ruff check . && uv run ruff format . && uv run pyright   # lint, format, strict types
uv run pre-commit install   # all of the above on every commit
```

Style: flat pytest functions, pyright strict, explicit ABCs, and no computer algebra in the code or its tests (exact
`Fraction` arithmetic where exactness is needed); the symbolic checks of the paper live with the paper.

## History

The original collocated code (2015; collocated centred derivatives, adaptive Dormand-Prince) lives on as `src/_old`. Its long-standing "high-frequency instability" was diagnosed in September 2026 as the odd-even null
mode of collocated centred first derivatives; the staggered layout of this code has no such mode, since the
sawtooth is its stiffest direction rather than an invisible one. The rebuild began on 2026-09-18.

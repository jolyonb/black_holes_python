# pbh-python code: map for new sessions

Owner: Jolyon Bloomfield (physicist, author). Spherically symmetric primordial-black-hole formation from a
perfect-fluid perturbation of flat FRW, Misner-Sharp formalism. This repo is the evolution code only; the theory,
numerics analysis and paper drafts live in the sibling `../analysis/` (its own git repo, not on GitHub). The
workspace map is `../CLAUDE.md`.

## Layout

| path | what | git |
|---|---|---|
| `src/pbh/` | **The production code, being built** (since 2026-09-18) from paper sections 7-8 (`../analysis/v4/numerics.tex`, `numerics-excision.tex`) in reviewed bites. Module docstrings name the paper section and equations they implement. Pipeline: `maps`/`geometry`/`layout`/`state` (grid and variables) -> `stencils`/`kernels`/`derived`/`equations` (one stage, in deviation form) -> `outer` (SAT closure) -> `timestep` (RK4, checked stepper, step rules) -> `horizon`/`excision` -> `driver` (the `Run`), with `config`, `initial`/`profiles`, `records`/`output`/`h5`, `monitors`, `readout`/`summary`, `michel`, `cli`. | tracked |
| `src/_old/` | **RETIRED** collocated code (the former `src/pbh`, moved 2026-09-18 with its unit tests deleted). Kept for reference, not run, not imported by `pbh`; still passes ruff and pyright strict. `ms.py` (Misner-Sharp EOMs, Eulerian and Lagrangian handlers), `base.py` (evolver + cached EOM handler), `derivs.py` (collocated stencils), `dopri5.py`, `initial.py` (2015 growing-mode data), `output.py`, `cli.py`. | tracked |
| `tests/` | Flat pytest functions, one file per module; fast suite by default (667), evolutions `-m slow` (27). `whole_state.py` is the stage as printed, the cross-check of the deviation form. | tracked |
| `README.md` | Still describes the OLD code's CLI and output; to be rewritten (the new CLI: `pbh validate`, `initial gaussian`, `run`, `restart`, `summary`). | tracked |
| `../analysis/` | **Outside this repo.** Theory and numerics rebuild plus the paper sources; see `../analysis/CLAUDE.md`. | sibling repo |

Note: the phase READMEs in `../analysis` were written when the code lived at `black_holes_python/` and ran under a
system Python 3.13. Here the retired code is `src/_old` and everything runs through uv.

## Running things

```
uv sync                      # Python 3.14; installs the dev and analysis groups (pytest, ruff, pyright, sympy, matplotlib)
uv run pytest                # fast suite
uv run pytest -m slow        # evolutions
uv run pytest -m ''          # everything
uv run ruff check . && uv run ruff format . && uv run pyright     # pre-commit runs all three
```

Analysis suites (from their own directory, using this repo's environment; macOS has no `timeout` command):

```
cd ../analysis/phaseA && uv run --project ../../code python -m pytest tests -q -c pytest.ini                # 127 tests, ~100 s
cd ../analysis/phaseB && uv run --project ../../code python -m pytest tests -q -c pytest.ini -m "not slow"  # 476 tests, ~60 s
```

## Conventions (do not relitigate; details in `../analysis/CLAUDE.md`)

* Variables: `r` = `Rtilde = R/(a R_H)`, `u` = `Utilde = U/(H a R_H)` (FRW value `r`), `m` = `mtilde` (FRW 1),
  `rho` = `rhotilde` (FRW 1), `xi = ln(t/t_0)`, `H = e^{-xi}`, `a = e^{alpha xi}`, `alpha = 2/(3(1+w))`, units `R_H = 1`.
  `w` is a rational parameter (default `1/3`; `alpha` and the rest derived in `eos.EquationOfState`); the exact
  outgoing-wave outer condition is `w = 1/3`-only and guarded.
* Slicing is Misner-Sharp throughout. No Hernandez-Misner, no lapse engineering. After horizon formation the plan is
  excision inside the apparent horizon on a grid pinned to areal radius near the hole.
* The paper is the spec and the code the implementation: a change to the scheme changes the paper in the same bite
  (analysis repo), and exploratory reports are not committed. The owner reads every line.
* Code style: flat pytest functions, ruff, pyright strict, explicit ABCs; no sympy in the code or its tests (exact
  `Fraction` arithmetic or numerics). Do not commit or push unless asked, each time.
* Verification per change: the fast suite, the slow suite when evolutions are touched, the check scripts of the
  affected sections; any `pbh` API change also runs `../analysis/v4/checks/sec7_numerics.py`, the only analysis
  script that imports `pbh`.

## Known numerics (why the rebuild exists)

The old code's "high-frequency instability in rho" is an odd-even sawtooth null mode of the collocated centred
first-derivative operators, pumped by variable coefficients. The old `tests/test_operators.py` pinned that defect and the
good `R^4` flux-form density operator (deleted 2026-09-18 with the old tests; the demonstration is rebuilt in the new
stencils bite). The structural fix is the staggered layout from Phase B.

In the retired code, the viscous lapse correction in `MSCommon.ephi` had the wrong sign until 2026-09-09 (fixed; effect was at most a 2%
lapse error at a shock, formation times unchanged to 5 digits). Output files record `w` (column 17) and restarts check it.

## Where things stand

See `../CLAUDE.md`, "Where things stand", for the bites done, the decisions of 2026-09-23/24 (theta-limiter,
chord-widened bounds, checked RK4 stepper, 5e-13 abort, RK4 only, o1 only, switch-on in areal radius) and the next
bites (31 `M_est` jitter, 33 density-floor switch, 22-26, 13c, 28).

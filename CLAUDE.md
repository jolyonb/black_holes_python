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
| `tests/` | Flat pytest functions, one file per module; fast suite by default (782 with the Rust engine), evolutions `-m slow` (40). `whole_state.py` is the stage as printed, the cross-check of the deviation form. | tracked |
| `benchmarks/` | `collapse.py`, the wall-clock benchmark of both engines (see Benchmarks below). | tracked |
| `README.md` | The production code: worked example, configuration, the evolution file, module map. | tracked |
| `../analysis/` | **Outside this repo.** Theory and numerics rebuild plus the paper sources; see `../analysis/CLAUDE.md`. | sibling repo |

Note: the phase READMEs in `../analysis` were written when the code lived at `black_holes_python/` and ran under a
system Python 3.13. Here the retired code is `src/_old` and everything runs through uv.

## Running things

```
uv sync                      # Python 3.14; installs the dev and analysis groups (pytest, ruff, pyright, sympy, matplotlib)
                             # pbh itself is pure Python. The Rust engine (package pbh-engine in rust/, maturin, Rust
                             # stable with clippy and rustfmt) is the OPT-IN group rust: `uv sync --group rust` and
                             # `uv run --group rust ...`; without it the Rust tests skip. The pre-commit gates pass it
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
* For development runs, set `output: {snapshots: milestones}` (initial state, formation, switch-on, end only): the
  steps then run free of snapshot times, about 2x faster at N = 400 and 1.25x at N = 1600 than the default schedule,
  with restart points kept where they matter. `none` keeps only the initial state.
* Verification per change: the fast suite, the slow suite when evolutions are touched, the check scripts of the
  affected sections; any `pbh` API change also runs `../analysis/v4/checks/sec7_numerics.py`, the only analysis
  script that imports `pbh`.

## Benchmarks (for catching a slowdown)

`uv run --group rust python benchmarks/collapse.py` times a supercritical collapse (Gaussian `A = 0.2`, `ell = 2`,
`Rtilde_max = 30`, sinh scale 3, `snapshots: milestones`) from its initial data to the mass read-out on both engines,
best of 3, and checks that the two engines' records are identical. Rerun it after a change that could cost time and
compare on the same machine, idle (the owner's machine is sometimes loaded, which moves timings 5-25 per cent).

2026-09-28, commit `becad1f`, Apple M1 Pro, macOS 15.5, Python 3.14.6:

|    N | steps | numpy | Rust | numpy per step | Rust per step | Rust faster | records |
|---:|---:|---:|---:|---:|---:|---:|---|
| 200 | 1798 | 1.93 s | 0.40 s | 1072 us | 224 us | 4.8x | identical |
| 400 | 3553 | 3.97 s | 0.89 s | 1118 us | 250 us | 4.5x | identical |
| 800 | 7080 | 9.17 s | 2.30 s | 1295 us | 325 us | 4.0x | identical |
| 1600 | 14090 | 21.69 s | 6.76 s | 1539 us | 480 us | 3.2x | identical |

The same collapse that morning, before the day's speed work: Rust 2.08 s at N = 400 and 12.81 s at N = 1600, numpy
4.38 s and 24.05 s.

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

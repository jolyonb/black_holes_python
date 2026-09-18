# pbh-python code: map for new sessions

Owner: Jolyon Bloomfield (physicist, author). Spherically symmetric primordial-black-hole formation from a
perfect-fluid perturbation of flat FRW, Misner-Sharp formalism. This repo is the evolution code only; the theory,
numerics analysis and paper drafts live in the sibling `../analysis/` (its own git repo, not on GitHub). The
workspace map is `../CLAUDE.md`.

## Layout

| path | what | git |
|---|---|---|
| `src/pbh/` | **The production code, being built** (since 2026-09-18) from paper sections 7-8 (`../analysis/v4/numerics.tex`, `numerics-excision.tex`) in reviewed bites, one module per subsection. Empty package until bite 1 lands. | tracked |
| `src/_old/` | **RETIRED** collocated code (the former `src/pbh`, moved 2026-09-18 with its unit tests deleted). Kept for reference, not run, not imported by `pbh`; still passes ruff and pyright strict. `ms.py` (Misner-Sharp EOMs, Eulerian and Lagrangian handlers), `base.py` (evolver + cached EOM handler), `derivs.py` (collocated stencils), `dopri5.py`, `initial.py` (2015 growing-mode data), `output.py`, `cli.py`. | tracked |
| `tests/` | Empty until bite 1. Flat pytest functions; fast suite by default, evolutions `-m slow`. | tracked |
| `README.md` | Describes the OLD code's CLI and output; to be rewritten when the new driver exists. | tracked |
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
  `w` is a parameter (rational, default `1/3`; derived constants cached on `EOMHandler`); the outer BC, the
  timeout and second-order initial data are `w = 1/3`-only and guarded.
* Slicing is Misner-Sharp throughout. No Hernandez-Misner, no lapse engineering. After horizon formation the plan is
  excision inside the apparent horizon on a grid pinned to areal radius near the hole.
* Refactors of the existing code should be bit-exact unless the commit says otherwise (goldens are the guard).
* Code style: flat pytest functions, ruff, pyright strict. Do not commit or push unless asked.

## Known numerics (why the rebuild exists)

The old code's "high-frequency instability in rho" is an odd-even sawtooth null mode of the collocated centred
first-derivative operators, pumped by variable coefficients. The old `tests/test_operators.py` pinned that defect and the
good `R^4` flux-form density operator (deleted 2026-09-18 with the old tests; the demonstration is rebuilt in the new
stencils bite). The structural fix is the staggered layout from Phase B.

The viscous lapse correction in `MSCommon.ephi` had the wrong sign until 2026-09-09 (fixed; effect was at most a 2%
lapse error at a shock, formation times unchanged to 5 digits). Output files record `w` (column 17) and restarts check it.

## Next steps, in order

1. (done 2026-09-09) `w` as a rational parameter; expression cleanup; viscous-lapse sign fix; `w` recorded in output.
2. **Theory paper first** (owner's decision 2026-09-09): revise `../analysis/updated/` section by section, the owner
   verifying each derivation, with the numerical-scheme section written before any of it is implemented. Plan in
   `../CLAUDE.md`.
3. **Phase C, IN PROGRESS (from 2026-09-18)**: the production code in `src/pbh`, written from paper sections 7-8 (the
   spec; `S_spec.tex` is out of date) in small owner-reviewed bites, geometry first. Each run is three files
   (`name.config.json`, `name.initial.h5`, `name.evolution.h5`; HDF5 via h5py, appendable and readable mid-run).
   Requirements: every line human-readable and read by the owner; a switch to continue a formed hole WITHOUT
   excision so the excised run can be shown to agree with it; the mass read as `M_est` (rate-corrected estimate).

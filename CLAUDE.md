# pbh-python: map for new sessions

Owner: Jolyon Bloomfield (physicist, author). Spherically symmetric primordial-black-hole formation from a
perfect-fluid perturbation of flat FRW, Misner-Sharp formalism. This file is the map; the detail lives where it points.

## Layout

| path | what | git |
|---|---|---|
| `src/pbh/` | The evolution code. `ms.py` (Misner-Sharp EOMs, Eulerian and Lagrangian handlers), `base.py` (evolver + cached EOM handler), `derivs.py` (sparse finite-difference stencils), `dopri5.py` (adaptive RK), `initial.py` (grid + growing-mode initial data), `output.py` (gnuplot `.dat` / `.npz` snapshots), `cli.py` (`pbh` entry point). | tracked |
| `tests/` | Flat pytest functions. Fast suite by default; full evolutions are `-m slow` (golden horizon-formation values live there). | tracked |
| `README.md` | User-facing: CLI usage, output columns, gnuplot recipes, Lagrangian-vs-Eulerian notes. | tracked |
| `analysis/` | **Local only, gitignored, never push.** Theory and numerics rebuild plus the paper sources. Start at `analysis/CLAUDE.md` (2-minute orientation), then `analysis/phaseA/README.md` (continuum theory, sympy-verified) and `analysis/phaseB/README.md` (discretisation, numpy-verified). The Phase C implementation spec is `analysis/phaseB/writeup/S_spec.tex`; prototype reference code is `analysis/phaseB/lib/`. | ignored |

Note: `analysis/CLAUDE.md` and the phase READMEs were written when the code lived at `black_holes_python/` and
ran under a system Python 3.13. Here the code is the root package `src/pbh` and everything runs through uv.

## Running things

```
uv sync                      # Python 3.14; installs the dev and analysis groups (pytest, ruff, pyright, sympy, matplotlib)
uv run pbh --help            # evolve; .dat or .npz output chosen by suffix; restart from a snapshot with `pbh run.npz --snapshot -1`
uv run pytest                # fast suite (~2 s)
uv run pytest -m slow        # full evolutions incl. goldens
uv run pytest -m ''          # everything
uv run ruff check . && uv run ruff format . && uv run pyright     # pre-commit runs all three
```

Analysis suites (from their own directory, using this repo's environment; macOS has no `timeout` command):

```
cd analysis/phaseA && uv run --project ../.. python -m pytest tests -q -c pytest.ini                  # 127 tests, ~100 s
cd analysis/phaseB && uv run --project ../.. python -m pytest tests -q -c pytest.ini -m "not slow"    # 476 tests, ~60 s
```

## Conventions (do not relitigate; details in `analysis/CLAUDE.md`)

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
first-derivative operators, pumped by variable coefficients. `tests/test_operators.py` pins both that defect and the
good `R^4` flux-form density operator. The structural fix is the staggered layout from Phase B.

Open (pre-existing, not fixed because it moves the viscous goldens; needs owner sign-off): the viscous lapse correction
in `MSCommon.ephi` enters with the wrong sign. With `x = w rho'/((1+w) rho) - P'/(P+rho)` and `xint(r) = -int_r^rmax x`,
`ephi_analytic * np.exp(-xint)` gives `d ln ephi/dr = -(2w/(1+w)) rho'/rho + P'/(P+rho)` instead of the Misner-Sharp
`-P'/(rho+P)`; with a forced `Q` the residual is exactly twice the correction (1.8e-2 vs 9e-3 at n = 1600), and
`np.exp(xint)` brings it to 6e-6. `analysis/updated/code.tex` prints the same flipped exponential and contradicts its own
`phi = -(w/(1+w)) ln rho - int_A^Amax [...]` two lines earlier. Fix = flip the sign, correct code.tex, re-record the
viscous goldens, add a forced-`Q` test of `d ln ephi/dr == -dPdr/(rho+P)`. Harmless with `Q = 0` (the split cancels).

## Next steps, in order

1. Expression cleanup (trades bit-identity for the cleanest forms; goldens at rel 1e-9 survive): one cached
   `H = exp(-xi)` and one cached `(H a)^2 = exp(2(alpha-1)xi)` at every site, replacing the mirrored spellings kept
   for bit-identity in the `w` refactor; fix the viscous-lapse sign above; record `w` in output files and check it on
   restart.
2. Phase C: implement `analysis/phaseB/writeup/S_spec.tex` as a new staggered core alongside the old handlers, with
   the acceptance tests in the spec's test table; then the production supercritical run (sigma = 2, amplitude 0.175)
   reading the mass from the enclosed-energy plateau.
3. Paper: revise `analysis/updated/` (drop Hernandez-Misner, add Eulerian/excision and numerical-scheme sections).

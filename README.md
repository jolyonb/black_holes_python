# Black Hole Evolution Code

This is the clean implementation of the black hole evolution code. Apart from being nicely structured and readable, it also links all physics equations in the code to equations in [this version of the paper](paper.pdf). Sometimes we have two versions of an equation coded, with one commented out. The reason for this is to reuse things that have already been computed, so as to cut down on computation. Note that we specialize to w=1/3 and alpha=1/2 (alpha is just a function of w). Also note that we take the only scale in the problem R_H=1. All dimensionful quantities can be reconstructed by reinserting factors of R_H (the horizon radius at the start of evolution). Unfortunately, I haven't finished updating the PDF file completely; there are a few issues towards the end of the file where I haven't yet propagated some redefinitions. Such issues are noted in the code where applicable.

## Getting started

The project is managed with [uv](https://docs.astral.sh/uv/) and requires Python 3.14.

```
uv sync                 # create .venv with runtime + dev dependencies
uv run pbh --help       # show all evolution options
uv run pbh              # evolve the default Gaussian perturbation, writing output.dat
uv run pbh -o run.dat --scheme lagrangian --gridpoints 1000 --amplitude 0.18
uv run pbh -o run.npz   # same, but as a numpy archive (one array per quantity, snapshot axis first)
uv run pbh run.npz --snapshot -1   # resume from the last snapshot of a previous run (.dat or .npz)
```

The command line entry point is `pbh.cli`, which uses an old algorithm to take a linearized \delta_m and construct the growing mode from it (`pbh.initial`). (We have better tools now.) The physics lives in `pbh.ms` (Misner-Sharp equations of motion, Eulerian and Lagrangian), built on `pbh.base` (generic evolver and cached equation-of-motion handler), `pbh.derivs` (finite difference stencils) and `pbh.dopri5` (adaptive Runge-Kutta integrator).

### Development

```
uv run pytest           # fast test suite (well under a second)
uv run pytest -m slow   # full evolutions, including golden regression tests for horizon formation
uv run pytest -m ''     # everything
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # strict type checking
uv run pre-commit install   # run all of the above automatically on each commit
```

I recommend using gnuplot to visualize the output (the output has been formatted according to gnuplot specifications). Some helpful plotting commands are listed in the readme file for this repository.


## Output formats

Two output formats are available, selected by the output file suffix. The gnuplot text format writes one block per
snapshot, with tab-separated columns as listed below. The `.npz` format stores the same quantities as numpy arrays
with a leading snapshot axis, which is much more convenient for analysis and comparison scripts:

```python
import numpy as np

with np.load("run.npz") as data:
    xi = data["xi"]  # shape (snapshots,)
    rho = data["rho"]  # shape (snapshots, gridpoints)
    index = data["index"]  # shape (gridpoints,)
```

## Plotting column numbers

The column numbers are as given; the column names are in parentheses.

1. Grid point number (index)
2. \tilde{R} (r)
3. \tilde{U} (u)
4. \tilde{M} (m)
5. \tilde{\rho} (rho)
6. R (rfull)
7. U (ufull)
8. M (mfull)
9. \rho (rhofull)
10. 2M/R (horizon)
11. Characteristic speed c_s^+ (in \tilde{R}) (cs+)
12. Characteristic speed c_s^- (in \tilde{R}) (cs-)
13. Fluid speed c_s^0 (in \tilde{R}) (cs0)
14. \xi (xi)
15. Q (Q)
16. e^\phi (ephi)


## Lagrangian Evolution

The Lagrangian evolution is characterized by having grid points move along with fluid elements. This means that the grid is continually changing. This is fine, except that when you have a large overdensity, the grid tends to fall towards that overdensity. This means that you tend to have a lot of grid points near the origin, then some very sparsely distributed points, then evenly distributed points once you get back into the cosmological regime. This is fine, until you want a high accuracy derivative in that sparsely distributed area. If a shock wave passes through this area, you're going to be in trouble, as the derivative quality is terrible there, and will lead to instabilities.

To diagnose this, try the following plots:

* Plot of grid point index as a function of radius
```
plot "output_lag.dat" ev :::2:: u 2:1 w p
```

* Plot of Q as a function of radius (this is where bad derivatives are really felt)
```
plot "output_lag.dat" ev :::2:: u 2:15 w p
```


## Eulerian Evolution

In the Eulerian evolution, the grid points stay at fixed radius. What this tends to mean is that you need to run a non-uniform grid so that you have resolution where you need it. If you're trying to resolve shocks, you may just need a lot of grid points everywhere!

Note that if you're getting integration errors from shocks, increasing resolution will usually help.

### Open problem: near-critical instability

I'm presently having trouble with:
(gridpoints=500, squeeze=2, Amax=10, amplitude=0.1737, sigma=2.0)

An earlier version of the code (see git history before September 2026) failed on this too, with viscosity=20 (so a lot of suppression), at 500, 1000 and 1500 gridpoints. In particular, it failed before a shock wave formed. It looked like a high frequency instability in rho was responsible (in a position without a huge amount of nonlinearity), suggesting that we had complex eigenvalues in our differentiation matrix? This needs testing carefully, and reproducing with the current code.

"""The radial profiles a perturbation can be specified by: the inputs to the initial data of `initial.py`.

Each family satisfies the `Profile` contract of `initial.py`, a vectorised function of the scaled radius; the
families here are the ones the paper uses, and `initial.py` does not know which it is given. A production datum takes
one as its seed (`initial.Seed`); the former recipe takes one as a profile at a start time.
"""

from dataclasses import dataclass

import numpy as np

from pbh.types import FloatArray


@dataclass(frozen=True)
class Gaussian:
    """The Gaussian `A exp(-X^2 / 2 ell^2)` as a mass profile: the paper's standard perturbation.

    It is meant for the mass, not the density: as the seed `delta_m0`, a Gaussian in the curvature profile `K` (the
    literature's `q = 1` family), or as `delta_m` at a start time. As a mass it is compensated (eq:lin:compensated) by
    construction, since the density `delta_m + X delta_m' / 3` then carries the underdense shell that balances the
    core; the same Gaussian given as `delta_rho` would leave its whole mass excess inside the box. Its compaction
    `X^2 delta_m0` peaks at `X = sqrt 2 ell` with the value `2 ell^2 A / e` (Section 5.4).
    """

    A: float
    ell: float

    def __call__(self, X: FloatArray) -> FloatArray:
        """The profile at the radii `X`."""
        return self.A * np.exp(-0.5 * (X / self.ell) ** 2)

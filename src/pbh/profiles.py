"""The radial profiles a perturbation can be specified by: the inputs to the initial data of `initial.py`.

Each family satisfies the `Profile` contract of `initial.py`, a vectorised function of the scaled areal radius;
the families here are the ones the paper uses, and `initial.py` does not know which it is given.
"""

from dataclasses import dataclass

import numpy as np

from pbh.types import FloatArray


@dataclass(frozen=True)
class Gaussian:
    """The Gaussian `A exp(-X^2 / 2 ell^2)` as a mass profile `delta_m`: the paper's standard perturbation.

    It is meant for the mass, not the density. As `delta_m` it is compensated (eq:lin:compensated) by construction,
    since the density `delta_m + X delta_m' / 3` then carries the underdense shell that balances the core; the same
    Gaussian given as `delta_rho` would leave its whole mass excess inside the box. At `xi = 0` its compaction
    `X^2 delta_m` peaks at `X = sqrt 2 ell` with the value `2 ell^2 A / e` (Section 5.4).
    """

    A: float
    ell: float

    def __call__(self, X: FloatArray) -> FloatArray:
        """The profile at the radii `X`."""
        return self.A * np.exp(-0.5 * (X / self.ell) ** 2)

"""RETIRED collocated code (2026-09-18): kept for reference, not run, superseded by the staggered package ``pbh``.

Misner-Sharp evolution of primordial black hole formation.

The main entry points are :class:`_old.ms.MS` (the evolver) together with one of the equation-of-motion handlers
:class:`_old.ms.MSEulerian` or :class:`_old.ms.MSLagrangian`. Initial data helpers live in :mod:`_old.initial`.
"""

from _old.base import BlackHoleEvolver, EOMHandler, EvolverError, Status
from _old.derivs import Derivative, DerivativeError
from _old.dopri5 import DOPRI5, DopriIntegrationError
from _old.ms import MS, MSCommon, MSEulerian, MSLagrangian

__all__ = [
    "DOPRI5",
    "MS",
    "BlackHoleEvolver",
    "Derivative",
    "DerivativeError",
    "DopriIntegrationError",
    "EOMHandler",
    "EvolverError",
    "MSCommon",
    "MSEulerian",
    "MSLagrangian",
    "Status",
]

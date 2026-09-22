"""The equation of state and the FRW background it fixes (paper Section 3, and the preamble of Section 7).

The fluid has `P = w rho` with `w` a constant, `0 < w <= 1`. Everything about the background follows from
`w` through the single exponent `alpha = 2 / (3 (1 + w))` of the scale factor `a = e^(alpha xi)`
(eq:asol, eq:axi), where `xi = ln(t / t_0)` is the time coordinate of the whole code. Units are `R_H = 1`, the
Hubble radius at `xi = 0`, so `H = e^(-xi)` (eq:Hxi).

`w` is a real number; the code represents it as an exact rational purely so that the radiation case works out
exactly: `w = 1/3` gives `alpha = 1/2` and the lapse exponent `-1/4` as the rationals they are, not as rounded
floats, and the run's configuration file records them without rounding.

This module holds every constant that depends on `w` alone, derived once and exactly, in `EquationOfState`, and the
FRW background at one time in `Background`: the scale factor, the Hubble rate and radius, the sound speed, the sound
horizon and the FRW value of `Gammabar^2`, evaluated once per stage by `Background.at(eos, xi)`. Nothing here knows
about grids or fields. Section 7 keeps every formula general in `w` except the outer closure, the energy
norm and the initial-data recipe, which exist for radiation only; `EquationOfState.is_radiation` is what those places
test.
"""

import math
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Self

import numpy as np

from pbh.types import FloatArray

#: The equation of state parameter of radiation, `w = 1/3`
RADIATION: Fraction = Fraction(1, 3)

#: Largest denominator a float `w` is rounded to when it is not already an exact rational.
_MAX_DENOMINATOR = 1_000_000


def as_rational_w(w: Fraction | int | float | str) -> Fraction:
    """Bring an equation of state parameter into the exact rational form the code uses, checking `0 < w <= 1`.

    A `Fraction`, an `int` or a string such as `"1/3"` or `"0.2"` is taken exactly. A float is rounded to the
    nearest rational with denominator at most `10**6`, with a warning when that changes its value, so that the float
    `1/3` means `Fraction(1, 3)`; pass a string or a `Fraction` when exactness matters.

    Args:
        w: The parameter of `P = w rho`.

    Returns:
        The same parameter as an exact `Fraction`.

    Raises:
        ValueError: If `w` is outside `0 < w <= 1` (dust and stiffer-than-light fluids are out of scope).
    """
    w_exact = Fraction(w).limit_denominator(_MAX_DENOMINATOR) if isinstance(w, float) else Fraction(w)
    if not 0 < w_exact <= 1:
        raise ValueError(f"w must satisfy 0 < w <= 1, got {w!r}")
    if isinstance(w, float) and float(w_exact) != w:
        warnings.warn(
            f"w = {w!r} was rounded to the rational {w_exact}; pass a Fraction or a string such as '1/3' for an exact "
            "value",
            stacklevel=2,
        )
    return w_exact


@dataclass(frozen=True)
class EquationOfState:
    """Everything that depends on `w` alone, derived once.

    Construct it as `EquationOfState(w)` with `w` in its exact rational representation (see `as_rational_w`);
    the remaining fields are filled in by `__post_init__` and are the constants the scheme uses in its hot loop, as
    floats. `w` and `alpha` are kept exact so that the run's configuration file can record them without rounding.

    Attributes:
        w: The equation of state parameter of `P = w rho`, in its exact rational representation.
        alpha: The scale-factor exponent `2 / (3 (1 + w))` (eq:asol), exact; `1/2` for radiation.
        sqrt_w: The sound speed of the fluid in units of light, `sqrt(w)` (eq:eul:speeds).
        lapse_exponent: The exponent of the algebraic lapse `e^phi = rhotilde ** lapse_exponent` for smooth flow,
            `-w / (1 + w) = -3 alpha w / 2` (eq:MSphinov); `-1/4` for radiation.
        energy_source_rate: The coefficient `2 - 3 alpha` of the energy balance law (eq:num:energy), which equals
            `3 alpha w` and is the rate at which the FRW cell content `Delta V_c` grows on a static comoving grid;
            `1/2` for radiation.
        growing_mode_rate: `lambda_g = 2 (1 - alpha)`, the growth rate in `xi` of the super-horizon growing mode
            (eq:num:stepcap); `1` for radiation.
        accretion_eigenvalue: `lambda_c = (1 + 3w)^((1 + 3w) / (2w)) / (4 w^(3/2))`, the Michel accretion rate in
            units of `n_inf M^2` (eq:exc:lambdac); `6 sqrt(3)` for radiation.
        sonic_radius_over_mass: `r_s / M = (1 + 3w) / (2w)`, the sonic radius of the Michel flow (eq:exc:sonic);
            `3` for radiation.
    """

    w: Fraction
    alpha: Fraction = field(init=False)
    sqrt_w: float = field(init=False)
    lapse_exponent: float = field(init=False)
    energy_source_rate: float = field(init=False)
    growing_mode_rate: float = field(init=False)
    accretion_eigenvalue: float = field(init=False)
    sonic_radius_over_mass: float = field(init=False)

    def __post_init__(self) -> None:
        """Derive the constants from `w`; the dataclass is frozen, so they are set through `object.__setattr__`."""
        if not 0 < self.w <= 1:
            raise ValueError(f"w must satisfy 0 < w <= 1, got {self.w}")
        w = self.w
        alpha = Fraction(2, 3) / (1 + w)
        one_plus_3w = 1 + 3 * w
        derived = {
            "alpha": alpha,
            "sqrt_w": math.sqrt(w),
            "lapse_exponent": float(-w / (1 + w)),
            "energy_source_rate": float(2 - 3 * alpha),
            "growing_mode_rate": float(2 * (1 - alpha)),
            "accretion_eigenvalue": float(one_plus_3w) ** float(one_plus_3w / (2 * w)) / (4 * float(w) ** 1.5),
            "sonic_radius_over_mass": float(one_plus_3w / (2 * w)),
        }
        for name, value in derived.items():
            object.__setattr__(self, name, value)

    def lapse(self, rho: FloatArray) -> FloatArray:
        """The algebraic lapse `e^phi = rho ** lapse_exponent` of a density (eq:MSphinov).

        numpy evaluates a non-integer power through a logarithm and an exponential; where the exponent allows, square
        roots do the same in a tenth of the time, correctly rounded at each step, so to within an ulp of the power.
        """
        if self.lapse_exponent == -0.25:  # radiation: rho^(-1/4)
            return 1.0 / np.sqrt(np.sqrt(rho))
        if self.lapse_exponent == -0.5:  # the stiff fluid, w = 1: rho^(-1/2)
            return 1.0 / np.sqrt(rho)
        return rho**self.lapse_exponent

    def lapse_and_deviation(self, rho: FloatArray, delta_rho: FloatArray) -> tuple[FloatArray, FloatArray]:
        """The lapse `e^phi` and its deviation `e^phi - 1` from its FRW value, the latter without subtracting one.

        Near FRW `e^phi - 1` is a small number, and forming it as `e^phi` minus one keeps only the rounding of `e^phi`.
        It is formed from `delta_rho = rho - 1` instead: for radiation, with `r = rho^(1/4)`,
        `e^phi - 1 = (1 - r) / r = -delta_rho / (r (1 + r) (1 + r^2))`, since `1 - r^4 = (1 - r)(1 + r)(1 + r^2)`; for
        the stiff fluid, with `r = rho^(1/2)`, `-delta_rho / (r (1 + r))`; for any other `w`,
        `expm1(lapse_exponent log1p(delta_rho))`.

        Args:
            rho: The density.
            delta_rho: Its deviation `rho - 1`, formed without subtracting one.
        """
        if self.lapse_exponent == -0.25:
            r2 = np.sqrt(rho)
            r = np.sqrt(r2)
            return 1.0 / r, -delta_rho / (r * (1.0 + r) * (1.0 + r2))
        if self.lapse_exponent == -0.5:
            r = np.sqrt(rho)
            return 1.0 / r, -delta_rho / (r * (1.0 + r))
        return rho**self.lapse_exponent, np.expm1(self.lapse_exponent * np.log1p(delta_rho))

    @property
    def is_radiation(self) -> bool:
        """Whether `w = 1/3`: the outer closure, the energy norm and the initial-data recipe exist only then."""
        return self.w == RADIATION


@dataclass(frozen=True)
class Background:
    """The FRW background at one time `xi`: the scalars every stage of the integrator needs, evaluated once.

    Build it with `Background.at(eos, xi)` at the top of a stage and pass it down, so that no exponential is re-derived
    at each place it is used and every consumer sees the same values. Units are `R_H = 1`.

    Attributes:
        xi: The time `xi = ln(t / t_0)` at which the background is evaluated.
        a: The scale factor `e^(alpha xi)` (eq:axi).
        H: The Hubble rate `e^(-xi)` (eq:Hxi); the same for every `w`.
        hubble_radius: The Hubble radius in the scaled coordinate, `Rtilde = 1 / (H a R_H) = e^((1 - alpha) xi)`
            (Section 5.1). It is also the FRW value of `Gammabar` (eq:newgamma with `Utilde = Rtilde`, `mtilde = 1`).
        Gammabar2: The FRW value of `Gammabar^2`, `e^(2 (1 - alpha) xi)`, the first term of `Gammabar_j^2` in
            eq:num:facefields.
        c_s: The sound speed `alpha sqrt(w) e^((1 - alpha) xi)` in `Rtilde` per unit `xi`: the signal speed `a` of
            eq:eul:speeds on FRW (Section 5.1); for radiation `c_s^2 = e^xi / 12`.
        tau: The sound horizon `c_s / (1 - alpha)`, the distance sound has travelled since `xi = -inf` (eq:lin:tau);
            for radiation `tau = 2 c_s = e^(xi / 2) / sqrt(3)`, and the Hubble radius is the fixed multiple
            `(1 + 3w) / (2 sqrt(w))` of it.
    """

    xi: float
    a: float
    H: float
    hubble_radius: float
    Gammabar2: float
    c_s: float
    tau: float

    @classmethod
    def at(cls, eos: EquationOfState, xi: float) -> Self:
        """Evaluate the FRW background of Section 3 at time `xi` for the fluid `eos`."""
        alpha = float(eos.alpha)
        hubble_radius = math.exp((1.0 - alpha) * xi)
        c_s = alpha * eos.sqrt_w * hubble_radius
        return cls(
            xi=xi,
            a=math.exp(alpha * xi),
            H=math.exp(-xi),
            hubble_radius=hubble_radius,
            Gammabar2=hubble_radius**2,
            c_s=c_s,
            tau=c_s / (1.0 - alpha),
        )

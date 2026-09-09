"""Base classes for handling PDE evolution for black holes.

Relies on the :mod:`pbh.dopri5` module to actually perform time evolution.
"""

import math
import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Self, overload

import numpy as np
from numpy import pi
from numpy.typing import NDArray

from pbh.dopri5 import DOPRI5, DopriIntegrationError
from pbh.output import Snapshot, SnapshotWriter

type FloatArray = NDArray[np.float64]
type Scalar = float | FloatArray
"""A quantity that is usually a single number, but may in principle vary from gridpoint to gridpoint."""
type EOSParameter = Fraction | int | float | str
"""The equation of state parameter w of P = w rho.

Accepted as a :class:`~fractions.Fraction`, an int, a decimal or rational string such as ``"1/3"`` or ``"0.2"``, or a
float (canonicalised to the nearest rational with denominator at most 10**6, so the float ``1/3`` means
``Fraction(1, 3)``).
"""

np.seterr(all="raise", under="ignore")

#: The equation of state parameter of radiation, w = 1/3: the default everywhere, and the only value for which the
#: outer boundary condition, the timeout heuristic and the second-order initial data exist.
RADIATION_W = Fraction(1, 3)


def as_rational_w(w: EOSParameter) -> Fraction:
    """Convert an equation of state parameter to an exact rational, checking that 0 < w <= 1."""
    wfrac = Fraction(w).limit_denominator(1_000_000) if isinstance(w, float) else Fraction(w)
    if not 0 < wfrac <= 1:
        # Only a float can differ from its rational form; say so, since the rounded value is what was rejected
        rounded = ""
        if isinstance(w, float) and wfrac != w:
            rounded = f" (a float is rounded to a rational with denominator at most 10**6: {wfrac})"
        raise ValueError(f"w must satisfy 0 < w <= 1, got {w!r}{rounded}")
    if isinstance(w, float) and float(wfrac) != w:
        warnings.warn(
            f"w = {w!r} was rounded to the rational {wfrac} (denominator at most 10**6); pass a Fraction or a string "
            f"such as '1/3' for an exact value",
            skip_file_prefixes=(os.path.dirname(os.path.abspath(__file__)),),  # attribute it to the caller of pbh
        )
    return wfrac


def alpha_of_w(w: Fraction) -> Fraction:
    """The exponent alpha = 2/(3(1+w)) of the scalefactor a = e^(alpha xi), exactly (Eq. (43b))."""
    return Fraction(2, 3) / (1 + w)


class Status(Enum):
    """Status of a :class:`BlackHoleEvolver` object."""

    # Descriptive statuses
    NEEDS_INITIALIZING = 0
    READY = 1
    TIMEOUT = 2
    BLACKHOLE_FORMED = 3
    # Error statuses
    INTEGRATION_ERROR = -1
    NEGATIVE_ENERGY_DENSITY = -2
    NEGATIVE_GAMMA2 = -3


class EvolverError(Exception):
    """Raised by an :class:`EOMHandler` when evolution cannot continue.

    Carries the :class:`Status` the evolver should adopt; the evolver catches the exception and updates its own status,
    so handlers never need a reference back to the evolver.
    """

    def __init__(self, status: Status, msg: str | None = None) -> None:
        self.status = status
        self.msg = msg or status.name
        super().__init__(self.msg)


class BlackHoleEvolver[H: EOMHandler]:
    """Generic evolution class for black holes.

    The class is generic in the type of :class:`EOMHandler` it drives.
    """

    def __init__(
        self,
        eomhandler: H,
        rtol: float = 1e-8,
        atol: float = 1e-8,
        cfl_safety: float = 0.75,
        debug: bool = False,
    ) -> None:
        """Initialize storage and prepare class for operation.

        Args:
            eomhandler: The equation of motion handler to drive.
            rtol: Relative tolerance for the integrator.
            atol: Absolute tolerance for the integrator.
            cfl_safety: Safety factor applied to the CFL step size limit.
            debug: Whether to print debugging information.
        """
        # Constants controlling integration
        self.rtol = rtol
        self.atol = atol
        self.cfl_safety = cfl_safety
        # Properties of what we're integrating (set by set_initial_conditions)
        self.gridpoints = 0
        self.index: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        # The equation of motion handler
        self.eomhandler: H = eomhandler
        # Other details
        self.debug = debug
        self.status = Status.NEEDS_INITIALIZING
        self.msg: str | None = None
        # The integrator is created by set_initial_conditions.
        self._integrator: DOPRI5 | None = None

    @property
    def integrator(self) -> DOPRI5:
        """The DOPRI5 integrator. Only available once initial conditions have been set."""
        if self._integrator is None:
            raise ValueError("Must call set_initial_conditions before accessing the integrator")
        return self._integrator

    def set_initial_conditions(self, start_xi: float, *start_fields: FloatArray) -> None:
        """Set initial conditions for time and the initial fields.

        Fields should be specified as individual fields passed as initial conditions.
        Assumes that each field has a value on each gridpoint.
        """
        # Initialize fields
        self.gridpoints = len(start_fields[0])
        for field in start_fields:
            if len(field) != self.gridpoints:
                raise ValueError("All fields must have the same number of gridpoints")
        self.index = np.arange(self.gridpoints)

        # Initialize the integrator
        self._integrator = DOPRI5(
            t0=start_xi,
            init_values=self.package_vars(*start_fields),
            derivs=self.derivatives,
            rtol=self.rtol,
            atol=self.atol,
        )

        # Initialize the EOM handler
        self.eomhandler.set_fields(self.integrator.t, self.integrator.values)
        self.eomhandler.initialize_derivatives()

        self.status = Status.READY

    def load_initial_conditions(self, filename: str | Path, snapshot: int = 0) -> None:
        """Load initial conditions from a snapshot of a data file written by :meth:`drive`.

        Both the gnuplot text format and the ``.npz`` format are understood (chosen by suffix). ``snapshot`` indexes
        the snapshots in the file; negative values count from the end, so ``-1`` resumes from the last one.
        """
        filename = Path(filename)
        if filename.suffix == ".npz":
            with np.load(filename) as archive:
                xi = float(archive["xi"][snapshot])
                r, u, m = archive["r"][snapshot], archive["u"][snapshot], archive["m"][snapshot]
                w = float(archive["w"][snapshot]) if "w" in archive else None
        else:
            xi, r, u, m, w = self._read_text_snapshot(filename, snapshot)
        if w is not None and w != self.eomhandler.w:
            raise ValueError(
                f"{filename} was written with w = {w!r}, but this evolver has w = {self.eomhandler.w!r} "
                f"({self.eomhandler.w_exact})"
            )
        self.set_initial_conditions(xi, r, u, m)

    @staticmethod
    def _read_text_snapshot(
        filename: Path, snapshot: int
    ) -> tuple[float, FloatArray, FloatArray, FloatArray, float | None]:
        """Read (xi, r, u, m, w) from one block of a gnuplot-format data file.

        Columns are located by the ``# name name ...`` header line of the block (see :meth:`snapshot`); ``w`` is
        None for files written before it was recorded.
        """
        blocks: list[tuple[list[str], list[list[str]]]] = []
        names: list[str] = []
        block: list[list[str]] = []
        with filename.open() as f:
            for raw_line in f:
                line = raw_line.strip()
                if line.startswith("#") and not block:
                    names = line[1:].split()  # the header line of the next block
                elif line.startswith("#"):
                    continue  # a comment inside a block
                elif line:
                    block.append(line.split("\t"))
                elif block:
                    blocks.append((names, block))
                    block = []
        if block:
            blocks.append((names, block))
        names, data = blocks[snapshot]
        if not names:
            raise ValueError(f"{filename} has no header line naming its columns")
        column = {name: i for i, name in enumerate(names)}
        for name in ("xi", "r", "u", "m"):
            if name not in column:
                raise ValueError(f"{filename} block {snapshot} has no column {name!r} (header: {names})")

        def col(name: str) -> FloatArray:
            return np.array([float(row[column[name]]) for row in data])

        xi = float(data[0][column["xi"]])
        w = float(data[0][column["w"]]) if "w" in column else None
        return xi, col("r"), col("u"), col("m"), w

    def evolve(self, stop_xi: float) -> bool:
        """Take steps forwards in time until the specified stop time.

        Returns:
            True if evolution was halted for any reason, or False if the step completed.
        """
        if self.status != Status.READY:
            raise ValueError(f"Class cannot evolve with status {self.status.name}")

        stepcount = 0
        while self.xi < stop_xi:
            stepcount += 1

            try:
                # Take a step
                self.integrator.step(stop_xi)

                # Perform post-processing
                if self.post_step_processing():
                    # Anything that gets here should have set the status appropriately
                    return True

                # Update CFL condition
                self.integrator.update_max_h(self.cfl_safety * self.cfl_check())
            except DopriIntegrationError as err:
                self._record_error(Status.INTEGRATION_ERROR, err.args[0])
                return True
            except EvolverError as err:
                self._record_error(err.status, err.msg)
                return True

        if self.debug:
            self.debug_evolve_complete(stepcount)

        return False

    def drive(
        self,
        output_step: float,
        writer: SnapshotWriter | None = None,
        max_time: float | None = None,
        write_after: float | None = None,
    ) -> None:
        """Run evolution, writing output periodically.

        A snapshot is passed to ``writer`` every ``output_step`` once the time is after ``write_after`` (no output is
        produced if ``writer`` is None). Stops if post processing requests it, or if ``max_time`` is reached.
        """
        if self.status != Status.READY:
            raise ValueError(f"Class cannot evolve with status {self.status.name}")

        # Write initial data
        if write_after is None or self.xi >= write_after:
            if self._output_or_abort(writer):
                return
            newtime = self.xi
        else:
            newtime = max(write_after, self.xi)
            if newtime == write_after:
                print(f"Evolving up to xi = {write_after} without output... This may take a little while.")

        # Integration loop
        while True:
            # Construct the time to integrate to
            while newtime <= self.xi:
                newtime += output_step

            # Take a step
            abort = self.evolve(newtime)

            # Write the data
            if (write_after is None or self.xi >= write_after) and self._output_or_abort(writer):
                return

            # Do we stop?
            if abort or self.post_output_processing():
                return
            if max_time is not None and self.xi >= max_time:
                self.status = Status.TIMEOUT
                return

    def _record_error(self, status: Status, msg: str) -> None:
        """Record that evolution has failed with the given status and message."""
        self.status = status
        self.msg = msg

    def _output_or_abort(self, writer: SnapshotWriter | None) -> bool:
        """Write a snapshot, returning True if the EOM handler found the state unphysical (status updated)."""
        if writer is None:
            return False
        try:
            writer.write(self.snapshot())
        except EvolverError as err:
            self._record_error(err.status, err.msg)
            return True
        return False

    @staticmethod
    def package_vars(*fields: FloatArray) -> FloatArray:
        """Take a list of fields and convert them into a single vector.

        e.g: [x, y, z], [v_x, v_y, v_z] -> [x, y, z, v_x, v_y, v_z]
        """
        return np.concatenate(fields)

    @property
    def xi(self) -> float:
        """The time the integrator is presently at."""
        return self.integrator.t

    def derivatives(self, xi: float, field_vec: FloatArray, params: object = None) -> FloatArray:
        """Compute the time derivative of ``field_vec`` at the given fields and time (DOPRI5 callback)."""
        del params  # Unused; part of the DOPRI5 callback signature
        # Set EOM handler to use the appropriate field values
        self.eomhandler.set_fields(xi, field_vec)
        return self.package_vars(*self.eomhandler.derivatives())

    def cfl_check(self) -> float:
        """Check the CFL condition and return the max step size allowed (not including a safety factor)."""
        # Set EOM handler to use the appropriate field values
        self.eomhandler.set_fields(self.integrator.t, self.integrator.values)
        return self.eomhandler.cfl_step()

    def snapshot(self) -> Snapshot:
        """Return the named quantities describing the current state of the system."""
        # Set EOM handler to use the appropriate field values
        self.eomhandler.set_fields(self.integrator.t, self.integrator.values)

        # Extract and name quantities in the order they'll appear in the data output
        return {  # gnuplot column
            "index": self.index,  # 1
            "r": self.eomhandler.r,  # 2
            "u": self.eomhandler.u,  # 3
            "m": self.eomhandler.m,  # 4
            "rho": self.eomhandler.rho,  # 5
            "rfull": self.eomhandler.rfull,  # 6
            "ufull": self.eomhandler.ufull,  # 7
            "mfull": self.eomhandler.mfull,  # 8
            "rhofull": self.eomhandler.rhofull,  # 9
            "horizon": self.eomhandler.horizon,  # 10
            "cs+": self.eomhandler.csp,  # 11
            "cs-": self.eomhandler.csm,  # 12
            "cs0": self.eomhandler.cs0,  # 13
            "xi": self.eomhandler.xi,  # 14
            "Q": self.eomhandler.Q,  # 15
            "ephi": self.eomhandler.ephi,  # 16
            "w": self.eomhandler.w,  # 17
        }

    # Optional methods

    def post_step_processing(self) -> bool:
        """Perform post processing after each step. Return True to stop evolution."""
        return False

    def post_output_processing(self) -> bool:
        """Perform post processing after output is written. Return True to stop evolution."""
        return False

    def debug_evolve_complete(self, stepcount: int) -> None:
        """Called at the completion of the evolve method if debug is turned on."""


class cached_property[T](property):  # noqa: N801 - mirrors functools.cached_property naming
    """Property caching its result in an :class:`EOMHandler`'s ``_cache`` dictionary.

    The cache is cleared whenever the handler's fields change (see :meth:`EOMHandler.set_fields`), so cached values
    are always consistent with the current state. Use this instead of the ``@property`` decorator for any quantity
    derived from the fields.
    """

    def __init__(self, func: Callable[[Any], T]) -> None:
        super().__init__(func, doc=func.__doc__)
        self._func = func
        self._name = func.__name__

    @overload
    def __get__(self, instance: None, owner: type | None = None, /) -> Self: ...

    @overload
    def __get__(self, instance: EOMHandler, owner: type | None = None, /) -> T: ...

    def __get__(self, instance: EOMHandler | None, owner: type | None = None, /) -> Self | T:
        if instance is None:
            return self
        cache = instance._cache  # pyright: ignore[reportPrivateUsage]
        if cache is None:
            raise ValueError("Must initialize EOMHandler with set_fields before requesting quantities")
        if self._name in cache:
            return cache[self._name]
        value = self._func(instance)
        cache[self._name] = value
        return value


class EOMHandler(ABC):
    """Generic equation of motion class, assuming evolved fields are r, u and m.

    * Handles all local computations.
    * Exposes methods for quantities needed by the evolver class.
    * Utilizes caching to ensure that quantities are not computed repeatedly.
    """

    #: Number of evolved fields (r, u and m)
    NUM_FIELDS = 3

    def __init__(self, viscosity: float | None = None, w: EOSParameter = RADIATION_W) -> None:
        """Initialize storage and operators.

        Args:
            viscosity: Artificial viscosity coefficient (None or 0 to disable).
            w: Equation of state parameter P = w rho (default 1/3, radiation). Rational; see :data:`EOSParameter`.
        """
        self.viscosity = viscosity
        # Equation of state constants, computed once in exact rational arithmetic and then floated. At w = 1/3 every
        # one of these floats is exact (inv_w = 3.0, alpha = 0.5, lapse_exponent = 0.25,
        # inv_cs_factor = sqrt(12)), which is what keeps the radiation numerics bitwise unchanged.
        #: The equation of state parameter w as an exact rational
        self.w_exact = as_rational_w(w)
        alpha_exact = alpha_of_w(self.w_exact)
        #: The equation of state parameter w
        self.w = float(self.w_exact)
        #: 1/w, so that P = rho / inv_w
        self.inv_w = float(1 / self.w_exact)
        #: alpha = 2/(3(1+w)), the exponent in a = e^(alpha xi) (Eq. (43b))
        self.alpha = float(alpha_exact)
        #: w/(1+w) = 3 alpha w/2, the exponent in e^phi = rho^(-w/(1+w)) (Eq. (45))
        self.lapse_exponent = float(self.w_exact / (1 + self.w_exact))
        #: 1/(alpha sqrt(w)), the inverse of the sound speed factor in the characteristic speed (Eq. (200))
        self.inv_cs_factor = math.sqrt(self.inv_w) / self.alpha
        #: Whether w = 1/3 exactly; gates the theory that exists only for radiation (outer boundary condition, timeout)
        self.is_radiation = self.w_exact == RADIATION_W
        # Initialize storage for state
        self._xi: Scalar | None = None
        self._fields: FloatArray | None = None
        # Initialize cache
        self._cache: dict[str, Any] | None = None

    def set_fields(self, xi: Scalar, fields: FloatArray) -> None:
        """Update the internal field values as needed. If the values are new, clear the cache."""
        if np.any(xi != self._xi) or np.any(fields != self._fields):
            # Update values and clear cache
            self._xi = xi if isinstance(xi, float | int) else xi.copy()
            self._fields = fields.copy()
            self._cache = {}

    @property
    def fields(self) -> FloatArray:
        """The current packed field vector."""
        if self._fields is None:
            raise ValueError("Must initialize EOMHandler with set_fields before requesting quantities")
        return self._fields

    @abstractmethod
    def initialize_derivatives(self) -> None:
        """Initialize any derivative operators. Called once initial conditions are known."""

    # Generic quantities: These quantities are independent of modelling approach; they do not require differentiation

    @property
    def xi(self) -> Scalar:
        """The time of the system. This may be different at different gridpoints."""
        if self._xi is None:
            raise ValueError("Must initialize EOMHandler with set_fields before requesting quantities")
        return self._xi

    @cached_property
    def a(self) -> Scalar:
        """The scalefactor a. This may be different at different gridpoints."""
        return np.exp(self.alpha * self.xi)  # Eq. (43b)

    @cached_property
    def H(self) -> Scalar:
        """The Hubble factor H = e^(-xi) (with R_H = 1). This may be different at different gridpoints."""
        return np.exp(-self.xi)  # Eq. (43c)

    @cached_property
    def Ha2(self) -> Scalar:
        """(H a R_H)^2 = e^(2(alpha-1)xi) = Gamma^2/Gammabar^2 (Eqs. (43b) and (43c)); equals H only for radiation."""
        return np.exp(2 * (self.alpha - 1) * self.xi)

    @cached_property
    def rho_b(self) -> Scalar:
        """The background density rho_b. This may be different at different gridpoints."""
        return 3 / 8 / pi * self.H * self.H  # Eq. (43e)

    @cached_property
    def horizon(self) -> FloatArray:
        """The apparent horizon condition 2M/R."""
        return self.r * self.r * self.m * self.Ha2  # Eq. (52): 2m/R = e^(2(alpha-1)xi) r^2 m

    @cached_property
    def gamma2(self) -> FloatArray:
        r"""\bar{\gamma}^2. Raises EvolverError with status NEGATIVE_GAMMA2 if negative anywhere."""
        gamma2 = 1 / self.Ha2 + self.u * self.u - self.r * self.r * self.m  # Eq. (44f): e^(2(1-alpha)xi) + u^2 - r^2 m
        if np.any(gamma2 < 0):
            raise EvolverError(Status.NEGATIVE_GAMMA2)
        return gamma2

    @cached_property
    def gamma(self) -> FloatArray:
        r"""\bar{\gamma}."""
        return np.sqrt(self.gamma2)  # By definition: sqrt(Gamma^2)

    @cached_property
    def rfull(self) -> FloatArray:
        """The physical radius R."""
        return self.a * self.r  # Eq. (41c)

    @cached_property
    def ufull(self) -> FloatArray:
        """The physical velocity U."""
        return self.a * self.H * self.u  # Eq. (41d)

    @cached_property
    def mfull(self) -> FloatArray:
        """The physical mass function m."""
        return 4 * pi / 3 * self.rho_b * self.rfull * self.rfull * self.rfull * self.m  # Eq. (41e)

    @cached_property
    def rhofull(self) -> FloatArray:
        """The physical density rho."""
        return self.rho_b * self.rho  # Eq. (41a)

    @cached_property
    def r(self) -> FloatArray:
        r"""\bar{R}."""
        return np.split(self.fields, self.NUM_FIELDS)[0]

    @cached_property
    def u(self) -> FloatArray:
        r"""\bar{U}."""
        return np.split(self.fields, self.NUM_FIELDS)[1]

    @cached_property
    def m(self) -> FloatArray:
        r"""\bar{m}."""
        return np.split(self.fields, self.NUM_FIELDS)[2]

    # Abstract quantities: These will need to be implemented on a case-by-case basis
    # Note that this is just the list of required properties; you can create others too!

    @property
    @abstractmethod
    def rho(self) -> FloatArray:
        r"""\bar{\rho}."""

    @property
    @abstractmethod
    def ephi(self) -> FloatArray:
        r"""e^\phi."""

    @property
    @abstractmethod
    def Q(self) -> FloatArray:
        r"""\bar{Q}, the artificial viscosity."""

    @property
    @abstractmethod
    def cs0(self) -> FloatArray:
        """The fluid speed c_0."""

    @property
    @abstractmethod
    def csp(self) -> FloatArray:
        """The speed of sound c_+."""

    @property
    @abstractmethod
    def csm(self) -> FloatArray:
        """The speed of sound c_-."""

    @abstractmethod
    def derivatives(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return a tuple of time derivatives for evolution (rdot, udot, mdot)."""

    @abstractmethod
    def cfl_step(self) -> float:
        """Return the maximum step size allowed by the CFL condition."""

"""Base classes for handling PDE evolution for black holes.

Relies on the :mod:`pbh.dopri5` module to actually perform time evolution.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, Self, TextIO, overload

import numpy as np
from numpy import pi
from numpy.typing import NDArray

from pbh.dopri5 import DOPRI5, DopriIntegrationError

type FloatArray = NDArray[np.float64]
type Scalar = float | FloatArray
"""A quantity that is usually a single number, but may in principle vary from gridpoint to gridpoint."""

np.seterr(all="raise", under="ignore")


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
        eomhandler: type[H],
        rtol: float = 1e-8,
        atol: float = 1e-8,
        cfl_safety: float = 0.75,
        viscosity: float | None = None,
        debug: bool = False,
    ) -> None:
        """Initialize storage and prepare class for operation.

        Args:
            eomhandler: The equation of motion handler class to instantiate.
            rtol: Relative tolerance for the integrator.
            atol: Absolute tolerance for the integrator.
            cfl_safety: Safety factor applied to the CFL step size limit.
            viscosity: Artificial viscosity coefficient (None or 0 to disable).
            debug: Whether to print debugging information.
        """
        # Constants controlling integration
        self.rtol = rtol
        self.atol = atol
        self.cfl_safety = cfl_safety
        # Properties of what we're integrating (set by set_initial_conditions)
        self.gridpoints = 0
        self.index: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        # Set up the equation of motion handler
        self.eomhandler: H = eomhandler(viscosity=viscosity)
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

    def load_initial_conditions(self, filename: str) -> None:
        """Load initial conditions from a data file.

        In particular, reads data from the first block of a data file in the format output by this class.
        """
        # Read the first block into an array
        data: list[list[str]] = []
        with open(filename) as f:
            while True:
                line = f.readline().strip()
                if line.startswith("#"):
                    # Ignore comments
                    continue
                if len(line) == 0:
                    # End of the first block
                    break
                data.append(line.split("\t"))

        # Grab the pieces we want from each line: r, u, m and xi
        r, u, m = np.array([[float(row[1]), float(row[2]), float(row[3])] for row in data]).transpose()
        xi = float(data[0][13])

        # Initialize everything
        self.set_initial_conditions(xi, r, u, m)

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
        file_handle: TextIO,
        max_time: float | None = None,
        write_after: float | None = None,
    ) -> None:
        """Run evolution, writing output periodically.

        Data is output every ``output_step`` to ``file_handle`` once the time is after ``write_after``.
        Stops if post processing requests it, or if ``max_time`` is reached.
        """
        if self.status != Status.READY:
            raise ValueError(f"Class cannot evolve with status {self.status.name}")

        # Write initial data
        if write_after is None or self.xi >= write_after:
            if self._output_or_abort(file_handle):
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
            if (write_after is None or self.xi >= write_after) and self._output_or_abort(file_handle):
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

    def _output_or_abort(self, file_handle: TextIO) -> bool:
        """Write output, returning True if the EOM handler found the state unphysical (status updated)."""
        try:
            self.output(file_handle)
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

    def output(self, file_handle: TextIO) -> None:
        """Output the current state of the system to the given file handle."""
        # Set EOM handler to use the appropriate field values
        self.eomhandler.set_fields(self.integrator.t, self.integrator.values)

        # Extract and name quantities in the order they'll appear in the data output
        data: dict[str, Scalar | NDArray[np.intp]] = {  # gnuplot column
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
        }

        # Write header
        file_handle.write("# " + "\t".join(data.keys()) + "\n")

        # Write the block of data
        for i in range(self.gridpoints):
            dat = [value[i] if isinstance(value, np.ndarray) else value for value in data.values()]
            file_handle.write("\t".join(map(str, dat)) + "\n")
        file_handle.write("\n")
        file_handle.flush()

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

    def __init__(self, viscosity: float | None = None) -> None:
        """Initialize storage and operators.

        Args:
            viscosity: Artificial viscosity coefficient (None or 0 to disable).
        """
        self.viscosity = viscosity
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
        return np.exp(self.xi / 2)  # Eq. (43b)

    @cached_property
    def H(self) -> Scalar:
        """The Hubble factor H (with R_H = 1). This may be different at different gridpoints."""
        return 1 / (self.a * self.a)  # Eqs. (43c) and (43b)
        # return np.exp(-self.xi)  # Eq. (43c)

    @cached_property
    def rho_b(self) -> Scalar:
        """The background density rho_b. This may be different at different gridpoints."""
        return 3 / 8 / pi * self.H * self.H  # Eq. (43e)

    @cached_property
    def horizon(self) -> FloatArray:
        """The apparent horizon condition 2M/R."""
        return self.r * self.r * self.m * self.H  # Eqs. (52) and (43c)
        # return self.r * self.r * self.m * np.exp(-self.xi)  # Eq. (52)

    @cached_property
    def gamma2(self) -> FloatArray:
        r"""\bar{\gamma}^2. Raises EvolverError with status NEGATIVE_GAMMA2 if negative anywhere."""
        gamma2 = 1 / self.H + self.u * self.u - self.r * self.r * self.m  # Eqs. (42f) and (43c)
        # gamma2 = np.exp(self.xi) + self.u * self.u - self.r * self.r * self.m  # Eq. (42f)
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

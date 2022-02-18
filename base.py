"""
base.py

Contains base classes for handling PDE evolution for black holes.

Relies on the dopri5 package to actually perform evolution.
"""

from __future__ import annotations

from abc import abstractmethod, ABC
import numpy as np
from numpy import pi
from enum import Enum
from typing import List, TextIO, Optional, Union, Tuple, Type
from numbers import Number
from collections import OrderedDict

from dopri5 import DOPRI5
from dopri853 import DOPRI853
from dopri_error import DopriIntegrationError

np.seterr(all='raise', under='ignore')


class EvolverError(Exception):
    """Generic exception caught as part of Evolver evolution"""


class Status(Enum):
    """Status of the BlackHoleEvolver object"""
    # Descriptive statuses
    NEEDS_INITIALIZING = 0
    READY = 1
    TIMEOUT = 2
    BLACKHOLE_FORMED = 3
    # Error statuses
    INTEGRATION_ERROR = -1
    NEGATIVE_ENERGY_DENSITY = -2
    NEGATIVE_GAMMA2 = -3


class BlackHoleEvolver(ABC):
    """
    Generic evolution class for black holes.
    """
    
    def __init__(self,
                 eomhandler: Type[EOMHandler],
                 rtol: float = 1e-8,
                 atol: float = 1e-8,
                 cfl_safety: float = 0.75,
                 viscosity: float = None,
                 debug: bool = False):
        """Initialize storage and prepare class for operation."""
        # Constants controlling integration
        self.rtol = rtol
        self.atol = atol
        self.cfl_safety = cfl_safety
        # Properties of what we're integrating
        self.num_fields = None
        self.gridpoints = None
        self.index = None
        # Store evolution quantities
        self.viscosity = viscosity
        self.eomhandler = eomhandler(parent=self)
        self.Qenvelope = None
        # Other details
        self.debug = debug
        self.status = Status.NEEDS_INITIALIZING
        self.msg = None
        # Set up the integrator. We initialize initial conditions later using set_initial_conditions.
        # self.integrator = DOPRI5(t0=None, init_values=None, derivs=self.derivatives, rtol=self.rtol, atol=self.atol)
        self.integrator = DOPRI853(t0=None, init_values=None, derivs=self.derivatives, rtol=self.rtol, atol=self.atol)

    def set_initial_conditions(self, start_xi: float, *start_fields: np.ndarray):
        """
        Sets initial conditions for time and the initial fields.
        Fields should be specified as individual fields passed as initial conditions.
        Assumes that each field has a value on each gridpoint.
        """
        # Initialize fields
        self.num_fields = len(start_fields)
        self.gridpoints = len(start_fields[0])
        for field in start_fields:
            assert len(field) == self.gridpoints
        self.index = np.array([i for i in range(self.gridpoints)])

        # Set up envelope for applying artificial viscosity - we don't want it applying at the boundary
        # We construct the envelope as a Fermi-Dirac distribution
        turnover_pt = float((self.gridpoints - 1) - 30)
        width = 3
        self.Qenvelope = 1 / (np.exp((self.index - turnover_pt) / width) + 1)

        # Initialize the integrator
        self.integrator.set_init_values(start_xi, self.package_vars(*start_fields))

        # Initialize the EOM handler
        self.eomhandler.set_fields(self.integrator.t, self.integrator.values)
        self.eomhandler.initialize_derivatives()

        self.status = Status.READY

    def load_initial_conditions(self, filename: str):
        """
        Load initial conditions from a data file.
        In particular, reads data from the first block of a data file in the format output by this class.
        """
        # Read the first block into an array
        data = []
        with open(filename) as f:
            while True:
                line = f.readline().strip()
                if line.startswith('#'):
                    # Ignore comments
                    continue
                if len(line) == 0:
                    # End of the first block
                    break
                data.append(line.split('\t'))

        # Grab the pieces we want from each line: r, u, m and xi
        r, u, m = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in data]).transpose()
        xi = float(data[0][13])
        
        # Initialize everything
        self.set_initial_conditions(xi, r, u, m)

    def evolve(self, stop_xi: float) -> bool:
        """
        Takes steps forwards in time until the specified stop time.
        Returns True if evolution was halted for any reason, or False if the step completed.
        """
        if self.status != Status.READY:
            raise ValueError(f'Class cannot evolve with status {self.status.name}')

        stepcount = 0
        while self.xi < stop_xi:
            stepcount += 1

            # Take a step
            try:
                self.integrator.step(stop_xi)
            except DopriIntegrationError as err:
                self.status = Status.INTEGRATION_ERROR
                self.msg = err.args[0]
                return True
            except EvolverError:
                # Anything that throws this error should update status appropriately before doing so
                return True

            # Perform post-processing
            if self.post_step_processing():
                # Anything that gets here should have set the status appropriately
                return True
            
            # Update CFL condition
            self.integrator.update_max_h(self.cfl_safety * self.cfl_check())

        if self.debug:
            self.debug_evolve_complete(stepcount)
            
        return False
    
    def drive(self,
              output_step: float,
              file_handle: TextIO,
              max_time: Optional[float] = None,
              write_after: Optional[float] = None,
              ):
        """
        Runs evolution, outputting data every output_step to the specified file_handle once the time is after
        write_after. Stops if post processing requests it, or if max_time is reached.
        """
        if self.status != Status.READY:
            raise ValueError(f'Class cannot evolve with status {self.status.name}')
    
        # Write initial data
        if write_after is None or self.xi >= write_after:
            self.output(file_handle)
            newtime = self.xi
        else:
            newtime = max(write_after, self.xi)
            if newtime == write_after:
                print(f'Evolving up to xi = {write_after} without output... This may take a little while.')

        # Integration loop
        while True:
            # Construct the time to integrate to
            while newtime <= self.xi:
                newtime += output_step

            # Take a step
            abort = self.evolve(newtime)

            # Write the data
            if write_after is None or self.xi >= write_after:
                self.output(file_handle)

            # Do we stop?
            if abort or self.post_output_processing():
                return
            if max_time is not None and self.xi >= max_time:
                self.status = Status.TIMEOUT
                return

    @staticmethod
    def package_vars(*fields: np.ndarray) -> np.ndarray:
        """
        Takes a list of fields and converts them into a single vector.
        e.g: [x, y, z], [v_x, v_y, v_z] -> [x, y, z, v_x, v_y, v_z]
        """
        return np.concatenate(fields)
    
    def unpackage_vars(self, field_vec: np.ndarray) -> List[np.ndarray]:
        """
        Takes a vector of fields and returns them as a list of fields.
        Note that the return values are views.
        e.g: [x, y, z, v_x, v_y, v_z] -> [x, y, z], [v_x, v_y, v_z]
        """
        return np.split(field_vec, self.num_fields)

    @property
    def xi(self) -> float:
        """Returns the time the integrator is presently at."""
        return self.integrator.t
    
    def derivatives(self, xi: float, field_vec: np.ndarray, params=None) -> np.ndarray:
        """Computes derivatives at the given value of the fields and t. Returns time derivative of field_vec."""
        # Set EOM handler to use the appropriate field values
        self.eomhandler.set_fields(xi, field_vec)
        return self.package_vars(*self.eomhandler.derivatives(params))

    def cfl_check(self) -> float:
        """Check the CFL condition and return the max step size allowed (not including a safety factor)."""
        # Set EOM handler to use the appropriate field values
        self.eomhandler.set_fields(self.integrator.t, self.integrator.values)
        return self.eomhandler.cfl_step()

    def output(self, file_handle: TextIO):
        """Outputs the current state of the system to the given file handle."""
        # Set EOM handler to use the appropriate field values
        self.eomhandler.set_fields(self.integrator.t, self.integrator.values)
        
        # Extract and name quantities in the order they'll appear in the data output
        data = OrderedDict([  # gnuplot index
            ('index', self.index),  # 1
            ('r', self.eomhandler.r),  # 2
            ('u', self.eomhandler.u),  # 3
            ('m', self.eomhandler.m),  # 4
            ('rho', self.eomhandler.rho),  # 5
            ('rfull', self.eomhandler.rfull),  # 6
            ('ufull', self.eomhandler.ufull),  # 7
            ('mfull', self.eomhandler.mfull),  # 8
            ('rhofull', self.eomhandler.rhofull),  # 9
            ('horizon', self.eomhandler.horizon),  # 10
            ('cs+', self.eomhandler.csp),  # 11
            ('cs-', self.eomhandler.csm),  # 12
            ('cs0', self.eomhandler.cs0),  # 13
            ('xi', self.eomhandler.xi),  # 14
            ('Q', self.eomhandler.Q),  # 15
            ('ephi', self.eomhandler.ephi),  # 16
        ])
        
        # Write header
        file_handle.write("# " + "\t".join(data.keys()) + "\n")
        
        # Write the block of data
        for i in range(self.gridpoints):
            dat = [data[field][i] if isinstance(data[field], np.ndarray) else data[field] for field in data]
            file_handle.write("\t".join(map(str, dat)) + "\n")
        file_handle.write("\n")
        file_handle.flush()

    # Optional methods

    def post_step_processing(self) -> bool:
        """
        Called after steps are completed to perform post processing.
        Return True to stop evolution.
        Optional method.
        """
        return False

    def post_output_processing(self) -> bool:
        """
        Called after output is written to determine whether to abort.
        Return True to stop evolution.
        Optional method.
        """
        return False

    def debug_evolve_complete(self, stepcount: int):
        """Called at the completion of the evolve method if debug is turned on."""


def cached_property(func):
    """
    Function decorator to cache results from an object's property using the object's _cache dictionary.
    Note that this should be used instead of the @property decorator.
    """
    
    def cache(self):
        funcname = func.__name__
        if self._cache is None:
            raise ValueError('Must initialize EOMHandler with set_fields before requesting quantities')
        ret_value = self._cache.get(funcname, None)
        if ret_value is None:
            ret_value = func(self)
            self._cache[funcname] = ret_value
        return ret_value
    
    return property(cache)


class EOMHandler(object):
    """
    Generic equation of motion class, assuming evolved fields are r, u and m.
    
    * Handles all local computations.
    * Exposes methods for quantities needed by the evolver class.
    * Utilizes caching to ensure that quantities are not computed repeatedly.
    """
    
    def __init__(self, parent: BlackHoleEvolver):
        """
        Initialize storage and operators.
        """
        # Store the parent object, so we can use some unpacking methods and update statuses
        self._parent = parent
        # Initialize storage for state
        self._xi = None
        self._fields = None
        # Initialize cache
        self._cache = None
    
    def set_fields(self, xi: Union[Number, np.ndarray], fields: np.ndarray):
        """
        Update the internal field values as needed.
        If the values are new, clear the cache.
        """
        if np.any(xi != self._xi) or np.any(fields != self._fields):
            # Update values and clear cache
            self._xi = xi if isinstance(xi, Number) else xi.copy()
            self._fields = fields.copy()
            self._cache = {}
    
    @abstractmethod
    def initialize_derivatives(self):
        """
        This abstract method is where any derivative operators should be initialized.
        """
        raise NotImplementedError()
    
    # Generic quantities: These quantities are independent of modelling approach; they do not require differentiation
    
    @property
    def xi(self) -> Union[Number, np.ndarray]:
        """Returns the time of the system. This may be different at different gridpoints."""
        return self._xi
    
    @cached_property
    def a(self) -> Union[Number, np.ndarray]:
        """Returns the scalefactor a. This may be different at different gridpoints."""
        return np.exp(self.xi / 2)  # Eq. (43b)
    
    @cached_property
    def H(self) -> Union[Number, np.ndarray]:
        """Returns the Hubble factor H (with R_H = 1). This may be different at different gridpoints."""
        return 1 / (self.a * self.a)  # Eqs. (43c) and (43b)
        # return np.exp(-self.xi)  # Eq. (43c)
    
    @cached_property
    def rho_b(self) -> Union[Number, np.ndarray]:
        """Returns the background density rho_b. This may be different at different gridpoints."""
        return 3 / 8 / pi * self.H * self.H  # Eq. (43e)
    
    @cached_property
    def horizon(self) -> np.ndarray:
        """Returns the apparent horizon condition."""
        return self.r * self.r * self.m * self.H  # Eqs. (52) and (43c)
        # return self.r * self.r * self.m * np.exp(-self.xi)  # Eq. (52)
    
    @cached_property
    def gamma2(self):
        r"""Returns \bar{\gamma}^2"""
        gamma2 = 1 / self.H + self.u * self.u - self.r * self.r * self.m  # Eqs. (42f) and (43c)
        # gamma2 = np.exp(self.xi) + self.u * self.u - self.r * self.r * self.m  # Eq. (42f)
        if np.any(gamma2 < 0):
            self._parent.status = Status.NEGATIVE_GAMMA2
            raise EvolverError()
        return gamma2

    @cached_property
    def gamma(self):
        r"""Returns \bar{\gamma}"""
        return np.sqrt(self.gamma2)  # By definition: sqrt(Gamma^2)

    @cached_property
    def rfull(self) -> np.ndarray:
        """Returns the physical radius R"""
        return self.a * self.r  # Eq. (41c)
    
    @cached_property
    def ufull(self) -> np.ndarray:
        """Returns the physical velocity U"""
        return self.a * self.H * self.u  # Eq. (41d)
    
    @cached_property
    def mfull(self) -> np.ndarray:
        """Returns the physical mass function m"""
        return 4 * pi / 3 * self.rho_b * self.rfull * self.rfull * self.rfull * self.m  # Eq. (41e)
    
    @cached_property
    def rhofull(self) -> np.ndarray:
        """Returns the physical density rho"""
        return self.rho_b * self.rho  # Eq. (41a)
    
    @cached_property
    def r(self) -> np.ndarray:
        """Returns \bar{R}"""
        self._extract_rum()
        return self._cache['r']
    
    @cached_property
    def u(self) -> np.ndarray:
        """Returns \bar{U}"""
        self._extract_rum()
        return self._cache['u']
    
    @cached_property
    def m(self) -> np.ndarray:
        """Returns \bar{m}"""
        self._extract_rum()
        return self._cache['m']
    
    def _extract_rum(self):
        """Extracts r, u and m from the field variables"""
        r, u, m = self._parent.unpackage_vars(self._fields)
        self._cache['r'] = r
        self._cache['u'] = u
        self._cache['m'] = m
    
    # Abstract quantities: These will need to be implemented on a case-by-case basis
    # Note that this is just the list of required properties; you can create others too!
    
    @property
    @abstractmethod
    def rho(self) -> np.ndarray:
        """Returns \bar{rho}"""
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def ephi(self) -> np.ndarray:
        """Returns e^phi"""
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def Q(self) -> np.ndarray:
        """Returns \bar{Q}"""
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def cs0(self) -> np.ndarray:
        """Returns the fluid speed c_0"""
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def csp(self) -> np.ndarray:
        """Returns speed of sound c_+"""
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def csm(self) -> np.ndarray:
        """Returns speed of sound c_-"""
        raise NotImplementedError()
    
    @abstractmethod
    def derivatives(self, params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns a tuple of time derivatives for evolution (rdot, udot, mdot)"""
        raise NotImplementedError()
    
    @abstractmethod
    def cfl_step(self) -> float:
        """Returns the maximum step size allowed by the CFL condition"""
        raise NotImplementedError()

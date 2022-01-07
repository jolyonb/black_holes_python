"""
baseclass.py

Contains the Evolver base class, used for evolving PDEs.
Relies on the dopri5 package to actually perform evolution.
"""
from abc import abstractmethod
import numpy as np
from typing import List, TextIO, Optional
from dopri5 import DOPRI5

class Evolver(object):
    """Abstract base class for performing PDE evolution."""
    
    def __init__(self, grid: np.ndarray, debug: bool = False):
        """
        Initialize storage. You should probably subclass this to accept parameters for evolution, and
        initialize things like the differentiator.
        """
        self.num_fields = None
        self.grid = grid
        self.gridpoints = len(grid)
        self.debug = debug
        # Set up the integrator. We initialize initial conditions later using set_initial_conditions.
        self.integrator = DOPRI5(t0=None, init_values=None, derivs=self.derivatives, rtol=1e-8, atol=1e-8)

    def set_initial_conditions(self, start_t: float, *start_fields: np.ndarray):
        """
        Sets initial conditions for time and the initial fields.
        Fields should be specified as individual fields passed as initial conditions.
        Assumes that each field has a value on each gridpoint.
        """
        for field in start_fields:
            assert len(field) == self.gridpoints
        self.integrator.set_init_values(start_t, self.package_vars(*start_fields))
        self.num_fields = len(start_fields)

    def evolve(self, stop_t: float) -> bool:
        """
        Takes steps forwards in time until the specified stop time.
        Returns True if post processing requested to stop evolution, or False if the step completed.
        """
        stepcount = 0
        while self.t < stop_t:
            stepcount += 1

            # Take a step
            self.integrator.step(stop_t)

            # Perform post-processing
            result = self.post_step_processing()
            if result:
                return True
            
            # Update CFL condition
            self.integrator.update_max_h(self.cfl_check())

        if self.debug:
            self.debug_evolve_complete(stepcount)
            
        return False
    
    def drive(self, output_step: float, file_handle: TextIO,
              max_time: Optional[float] = None, write_after: Optional[float] = None):
        """
        Runs evolution, outputting data every output_step to the specified file_handle once the time is after
        write_after. Stops if post processing requests it, or if max_time is reached.
        """
        # Write initial data to outfile
        if self.t >= write_after:
            self.output(file_handle)

        # Integration loop
        newtime = self.t
        while True:
            # Construct the time to integrate to
            while newtime <= self.t:
                newtime += output_step

            # Take a step
            abort = self.evolve(newtime)

            # Write the data
            if self.t >= write_after:
                self.output(file_handle)

            # Do we stop?
            if abort or (max_time is not None and self.t >= max_time) or self.post_output_processing():
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
    def fields(self) -> List[np.ndarray]:
        """
        Returns the list of fields currently stored in the integrator.
        """
        return self.unpackage_vars(self.integrator.values)
    
    @property
    def t(self) -> float:
        """
        Returns the time the integrator is presently at.
        """
        return self.integrator.t
    
    # Abstract methods - to be implemented!

    @abstractmethod
    def derivatives(self, t, field_vec, params=None) -> np.ndarray:
        """Computes derivatives at the given value of the fields and t. Returns time derivative of field_vec."""
        raise NotImplementedError()

    @abstractmethod
    def cfl_check(self) -> float:
        """Check the CFL condition and return the max step size allowed."""
        raise NotImplementedError()

    @abstractmethod
    def output(self, file_handle: TextIO):
        """Outputs the current state of the system to the given file handle."""
        raise NotImplementedError()

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
        pass

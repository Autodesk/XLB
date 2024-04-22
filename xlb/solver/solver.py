# Base class for all stepper operators

from xlb.compute_backend import ComputeBackend
from xlb.default_config import DefaultConfig
from xlb.operator.operator import Operator


class Solver(Operator):
    """
    Abstract class for the construction of lattice boltzmann solvers
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        boundary_conditions=[],
        velocity_set=None,
        precision_policy=None,
        compute_backend=None,
        grid_backend=None,
        grid_configs={},
    ):

        # Set parameters
        self.shape = shape
        self.velocity_set = velocity_set or DefaultConfig.velocity_set
        self.precision_policy = precision_policy or DefaultConfig.precision_policy
        self.compute_backend = compute_backend or DefaultConfig.compute_backend
        self.grid_backend = grid_backend or DefaultConfig.grid_backend
        self.boundary_conditions = boundary_conditions

        # Make grid
        if self.grid_backend is GridBackend.JAX:
            self.grid = JaxGrid(**grid_configs)
        elif self.grid_backend is GridBackend.WARP:
            self.grid = WarpGrid(**grid_configs)
        elif self.grid_backend is GridBackend.OOC:
            self.grid = OOCGrid(**grid_configs)
        else:
            raise ValueError(f"Grid backend {self.grid_backend} not recognized")

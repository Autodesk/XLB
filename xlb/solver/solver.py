from xlb.default_config import DefaultConfig
from xlb.operator.operator import Operator


class Solver(Operator):
    """
    Abstract class for the construction of lattice boltzmann solvers
    """

    def __init__(
        self,
        domain_shape: tuple[int, int, int],
        boundary_conditions=[],
        velocity_set=None,
        precision_policy=None,
        compute_backend=None,
    ):
        # Set parameters
        self.domain_shape = domain_shape
        self.boundary_conditions = boundary_conditions
        self.velocity_set = velocity_set or DefaultConfig.velocity_set
        self.precision_policy = precision_policy or DefaultConfig.precision_policy
        self.compute_backend = compute_backend or DefaultConfig.compute_backend

# Base class for all stepper operators

from xlb.compute_backends import ComputeBackends
from xlb.operator.boundary_condition import ImplementationStep
from xlb.global_config import GlobalConfig
from xlb.operator import Operator


class Solver(Operator):
    """
    Abstract class for the construction of lattice boltzmann solvers
    """

    def __init__(
        self,
        velocity_set=None,
        compute_backend=None,
        precision_policy=None,
        boundary_conditions=[],
    ):
        # Set parameters
        self.velocity_set = velocity_set or GlobalConfig.velocity_set
        self.compute_backend = compute_backend or GlobalConfig.compute_backend
        self.precision_policy = precision_policy or GlobalConfig.precision_policy
        self.boundary_conditions = boundary_conditions

        # Get collision and stream boundary conditions
        self.collision_boundary_conditions = {}
        self.stream_boundary_conditions = {}
        for id_number, bc in enumerate(self.boundary_conditions):
            bc_id = id_number + 1
            if bc.implementation_step == ImplementationStep.COLLISION:
                self.collision_boundary_conditions[bc_id] = bc
            elif bc.implementation_step == ImplementationStep.STREAMING:
                self.stream_boundary_conditions[bc_id] = bc
            else:
                raise ValueError("Boundary condition step not recognized")

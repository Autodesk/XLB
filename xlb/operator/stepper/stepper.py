# Base class for all stepper operators
from xlb.operator import Operator
from xlb import DefaultConfig


class Stepper(Operator):
    """
    Class that handles the construction of lattice boltzmann stepping operator
    """

    def __init__(self, operators, boundary_conditions):
        # Get the boundary condition ids
        from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry

        self.operators = operators
        self.boundary_conditions = boundary_conditions

        # Get velocity set, precision policy, and compute backend
        velocity_sets = set([op.velocity_set for op in self.operators if op is not None])
        assert len(velocity_sets) < 2, "All velocity sets must be the same. Got {}".format(velocity_sets)
        velocity_set = DefaultConfig.velocity_set if not velocity_sets else velocity_sets.pop()

        precision_policies = set([op.precision_policy for op in self.operators if op is not None])
        assert len(precision_policies) < 2, "All precision policies must be the same. Got {}".format(precision_policies)
        precision_policy = DefaultConfig.default_precision_policy if not precision_policies else precision_policies.pop()

        compute_backends = set([op.compute_backend for op in self.operators if op is not None])
        assert len(compute_backends) < 2, "All compute backends must be the same. Got {}".format(compute_backends)
        compute_backend = DefaultConfig.default_backend if not compute_backends else compute_backends.pop()

        # Initialize operator
        super().__init__(velocity_set, precision_policy, compute_backend)

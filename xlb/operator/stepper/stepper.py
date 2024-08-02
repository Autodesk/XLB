# Base class for all stepper operators
from xlb.operator import Operator
from xlb import DefaultConfig


class Stepper(Operator):
    """
    Class that handles the construction of lattice boltzmann stepping operator
    """

    def __init__(self, operators, boundary_conditions):
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

        # Add boundary conditions
        ############################################
        # Warp cannot handle lists of functions currently
        # TODO: Fix this later
        ############################################
        from xlb.operator.boundary_condition.bc_equilibrium import EquilibriumBC
        from xlb.operator.boundary_condition.bc_do_nothing import DoNothingBC
        from xlb.operator.boundary_condition.bc_halfway_bounce_back import HalfwayBounceBackBC
        from xlb.operator.boundary_condition.bc_fullway_bounce_back import FullwayBounceBackBC


        # Define a list of tuples with attribute names and their corresponding classes
        conditions = [
            ("equilibrium_bc", EquilibriumBC),
            ("do_nothing_bc", DoNothingBC),
            ("halfway_bounce_back_bc", HalfwayBounceBackBC),
            ("fullway_bounce_back_bc", FullwayBounceBackBC),
        ]

        # this fall-back BC is just to ensure Warp codegen does not produce error when a particular BC is not used in an example.
        bc_fallback = boundary_conditions[0]

        # Iterate over each boundary condition
        for attr_name, bc_class in conditions:
            for bc in boundary_conditions:
                if isinstance(bc, bc_class):
                    setattr(self, attr_name, bc)
                    break
                elif not hasattr(self, attr_name):
                    setattr(self, attr_name, bc_fallback)


        ############################################

        # Initialize operator
        super().__init__(velocity_set, precision_policy, compute_backend)

# Base class for all stepper operators

from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp

from xlb.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.precision_caster import PrecisionCaster


class Stepper(Operator):
    """
    Class that handles the construction of lattice boltzmann stepping operator
    """

    def __init__(
        self,
        collision,
        stream,
        equilibrium,
        macroscopic,
        boundary_conditions=[],
        forcing=None,  # TODO: Add forcing later
    ):
        # Add operators
        self.collision = collision
        self.stream = stream
        self.equilibrium = equilibrium
        self.macroscopic = macroscopic
        self.boundary_conditions = boundary_conditions
        self.forcing = forcing

        # Get all operators for checking
        self.operators = [
            collision,
            stream,
            equilibrium,
            macroscopic,
            *self.boundary_conditions,
        ]
        if forcing is not None:
            self.operators.append(forcing)

        # Get velocity set, precision policy, and compute backend
        velocity_sets = set([op.velocity_set for op in self.operators])
        assert len(velocity_sets) == 1, "All velocity sets must be the same"
        velocity_set = velocity_sets.pop()
        precision_policies = set([op.precision_policy for op in self.operators])
        assert len(precision_policies) == 1, "All precision policies must be the same"
        precision_policy = precision_policies.pop()
        compute_backends = set([op.compute_backend for op in self.operators])
        assert len(compute_backends) == 1, "All compute backends must be the same"
        compute_backend = compute_backends.pop()

        # Add boundary conditions
        # Warp cannot handle lists of functions currently
        # Because of this we manually unpack the boundary conditions
        ############################################
        # TODO: Fix this later
        ############################################
        from xlb.operator.boundary_condition.equilibrium import EquilibriumBC
        from xlb.operator.boundary_condition.do_nothing import DoNothingBC
        from xlb.operator.boundary_condition.halfway_bounce_back import HalfwayBounceBackBC
        from xlb.operator.boundary_condition.fullway_bounce_back import FullwayBounceBackBC
        self.equilibrium_bc = None
        self.do_nothing_bc = None
        self.halfway_bounce_back_bc = None
        self.fullway_bounce_back_bc = None
        for bc in boundary_conditions:
            if isinstance(bc, EquilibriumBC):
                self.equilibrium_bc = bc
            elif isinstance(bc, DoNothingBC):
                self.do_nothing_bc = bc
            elif isinstance(bc, HalfwayBounceBackBC):
                self.halfway_bounce_back_bc = bc
            elif isinstance(bc, FullwayBounceBackBC):
                self.fullway_bounce_back_bc = bc
        if self.equilibrium_bc is None:
            self.equilibrium_bc = EquilibriumBC(
                rho=1.0,
                u=(0.0, 0.0, 0.0),
                equilibrium_operator=self.equilibrium,
                velocity_set=velocity_set,
                precision_policy=precision_policy,
                compute_backend=compute_backend
            )
        if self.do_nothing_bc is None:
            self.do_nothing_bc = DoNothingBC(
                velocity_set=velocity_set,
                precision_policy=precision_policy,
                compute_backend=compute_backend
            )
        if self.halfway_bounce_back_bc is None:
            self.halfway_bounce_back_bc = HalfwayBounceBackBC(
                velocity_set=velocity_set,
                precision_policy=precision_policy,
                compute_backend=compute_backend
            )
        if self.fullway_bounce_back_bc is None:
            self.fullway_bounce_back_bc = FullwayBounceBackBC(
                velocity_set=velocity_set,
                precision_policy=precision_policy,
                compute_backend=compute_backend
            )
        ############################################

        # Initialize operator
        super().__init__(velocity_set, precision_policy, compute_backend)

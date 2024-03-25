# Base class for all stepper operators

from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp

from xlb.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
#from xlb.operator.boundary_condition.boundary_condition import ImplementationStep
#from xlb.operator.boundary_condition.boundary_applier.collision_boundary_applier import CollisionBoundaryApplier
#from xlb.operator.boundary_condition.boundary_applier.stream_boundary_applier import StreamBoundaryApplier
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
        equilibrium_bc,
        do_nothing_bc,
        half_way_bc,
        forcing=None,
    ):
        # Set parameters
        self.collision = collision
        self.stream = stream
        self.equilibrium = equilibrium
        self.macroscopic = macroscopic
        self.equilibrium_bc = equilibrium_bc
        self.do_nothing_bc = do_nothing_bc
        self.half_way_bc = half_way_bc
        self.boundary_conditions = [
            equilibrium_bc,
            do_nothing_bc,
            half_way_bc,
        ]
        self.forcing = forcing

        # Get all operators for checking
        self.operators = [
            collision,
            stream,
            equilibrium,
            macroscopic,
            *self.boundary_conditions,
            #forcing,
        ]

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

        # Make single operators for all collision and streaming boundary conditions
        #self.collision_boundary_applier = CollisionBoundaryApplier(
        #    [bc.boundary_applier for bc in boundary_conditions if bc.implementation_step == ImplementationStep.COLLISION]
        #)
        #self.streaming_boundary_applier = StreamBoundaryApplier(
        #    [bc.boundary_applier for bc in boundary_conditions if bc.implementation_step == ImplementationStep.STREAMING]
        #)

        # Initialize operator
        super().__init__(velocity_set, precision_policy, compute_backend)

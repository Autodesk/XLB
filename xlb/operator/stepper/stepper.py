# Base class for all stepper operators

from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp

from xlb.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.boundary_condition import ImplementationStep
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
        forcing=None,
    ):
        # Set parameters
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
            *boundary_conditions,
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

        # Make operators for converting the precisions
        #self.cast_to_compute = PrecisionCaster(

        # Make operator for setting boundary condition arrays
        self.set_boundary = SetBoundary(
            self.collision_boundary_conditions,
            self.stream_boundary_conditions,
            velocity_set,
            precision_policy,
            compute_backend,
        )
        self.operators.append(self.set_boundary)

        # Initialize operator
        super().__init__(velocity_set, precision_policy, compute_backend)


class SetBoundary(Operator):
    """
    Class that handles the construction of lattice boltzmann boundary condition operator
    This will probably never be used directly and it might be better to refactor it
    """

    def __init__(
        self,
        collision_boundary_conditions,
        stream_boundary_conditions,
        velocity_set,
        precision_policy,
        compute_backend,
    ):
        super().__init__(velocity_set, precision_policy, compute_backend)

        # Set parameters
        self.collision_boundary_conditions = collision_boundary_conditions
        self.stream_boundary_conditions = stream_boundary_conditions

    def _apply_all_bc(self, ijk, boundary_id, mask, bc):
        """
        Apply all boundary conditions
        """
        for id_number, bc in self.collision_boundary_conditions.items():
            boundary_id, mask = bc.boundary_masker(ijk, boundary_id, mask, id_number)
        for id_number, bc in self.stream_boundary_conditions.items():
            boundary_id, mask = bc.boundary_masker(ijk, boundary_id, mask, id_number)
        return boundary_id, mask

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, ijk):
        """
        Set boundary condition arrays
        These store the boundary condition information for each boundary
        """
        boundary_id = jnp.zeros(ijk.shape[:-1], dtype=jnp.uint8)
        mask = jnp.zeros(ijk.shape[:-1] + (self.velocity_set.q,), dtype=jnp.bool_)
        return self._apply_all_bc(ijk, boundary_id, mask, bc)

    @Operator.register_backend(ComputeBackend.PALLAS)
    def pallas_implementation(self, ijk):
        """
        Set boundary condition arrays
        These store the boundary condition information for each boundary
        """
        raise NotImplementedError("Pallas implementation not available")

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, ijk):
        """
        Set boundary condition arrays
        These store the boundary condition information for each boundary
        """
        boundary_id = wp.zeros(ijk.shape[:-1], dtype=wp.uint8)
        mask = wp.zeros(ijk.shape[:-1] + (self.velocity_set.q,), dtype=wp.bool)
        return self._apply_all_bc(ijk, boundary_id, mask, bc)

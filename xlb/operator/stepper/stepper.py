# Base class for all stepper operators

import jax.numpy as jnp

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.boundary_condition import ImplementationStep


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
        precision_policy=None,
        compute_backend=ComputeBackend.JAX,
    ):
        # Set parameters
        self.collision = collision
        self.stream = stream
        self.equilibrium = equilibrium
        self.macroscopic = macroscopic
        self.boundary_conditions = boundary_conditions
        self.forcing = forcing
        self.precision_policy = precision_policy
        self.compute_backend = compute_backend

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

        # Get all operators for checking
        self.operators = [
            collision,
            stream,
            equilibrium,
            macroscopic,
            *boundary_conditions,
        ]

        # Get velocity set and backend
        velocity_sets = set([op.velocity_set for op in self.operators])
        assert len(velocity_sets) == 1, "All velocity sets must be the same"
        self.velocity_set = velocity_sets.pop()
        compute_backends = set([op.compute_backend for op in self.operators])
        assert len(compute_backends) == 1, "All compute backends must be the same"
        self.compute_backend = compute_backends.pop()

        # Initialize operator
        super().__init__(self.velocity_set, self.compute_backend)

    def set_boundary(self, ijk):
        """
        Set boundary condition arrays
        These store the boundary condition information for each boundary
        """
        # Empty boundary condition array
        boundary_id = jnp.zeros(ijk.shape[:-1], dtype=jnp.uint8)
        mask = jnp.zeros(ijk.shape[:-1] + (self.velocity_set.q,), dtype=jnp.bool_)

        # Set boundary condition arrays
        for id_number, bc in self.collision_boundary_conditions.items():
            boundary_id, mask = bc.set_boundary(ijk, boundary_id, mask, id_number)
        for id_number, bc in self.stream_boundary_conditions.items():
            boundary_id, mask = bc.set_boundary(ijk, boundary_id, mask, id_number)

        return boundary_id, mask

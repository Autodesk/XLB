"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit, device_count
from functools import partial
import numpy as np
from enum import Enum

from xlb.operator.operator import Operator
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backends import ComputeBackends

# Enum for implementation step
class ImplementationStep(Enum):
    COLLISION = 1
    STREAMING = 2

class BoundaryCondition(Operator):
    """
    Base class for boundary conditions in a LBM simulation.
    """

    def __init__(
            self,
            set_boundary,
            implementation_step: ImplementationStep,
            velocity_set: VelocitySet,
            compute_backend: ComputeBackends.JAX,
        ):
        super().__init__(velocity_set, compute_backend)

        # Set implementation step
        self.implementation_step = implementation_step

        # Set boundary function
        if compute_backend == ComputeBackends.JAX:
            self.set_boundary = set_boundary
        else:
            raise NotImplementedError

    @classmethod
    def from_indices(cls, indices, implementation_step: ImplementationStep):
        """
        Creates a boundary condition from a list of indices.
        """
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def apply_jax(self, f_pre, f_post, mask, velocity_set: VelocitySet):
        """
        Applies the boundary condition.
        """
        pass

    @staticmethod
    def _indices_to_tuple(indices):
        """
        Converts a tensor of indices to a tuple for indexing
        TODO: Might be better to index
        """
        return tuple([indices[:, i] for i in range(indices.shape[1])])

    @staticmethod
    def _set_boundary_from_indices(indices):
        """
        This create the standard set_boundary function from a list of indices.
        `boundary_id` is set to `id_number` at the indices and `mask` is set to `True` at the indices.
        Many boundary conditions can be created from this function however some may require a custom function such as
        HalfwayBounceBack.
        """

        # Create a mask function
        def set_boundary(ijk, boundary_id, mask, id_number):
            """
            Sets the mask id for the boundary condition.

            Parameters
            ----------
            ijk : jnp.ndarray
                Array of shape (N, N, N, 3) containing the meshgrid of lattice points.
            boundary_id : jnp.ndarray
                Array of shape (N, N, N) containing the boundary id. This will be modified in place and returned.
            mask : jnp.ndarray
                Array of shape (N, N, N, Q) containing the mask. This will be modified in place and returned.
            """

            # Get local indices from the meshgrid and the indices
            local_indices = ijk[BoundaryCondition._indices_to_tuple(indices)]

            # Set the boundary id
            boundary_id = boundary_id.at[BoundaryCondition._indices_to_tuple(indices)].set(id_number)

            # Set the mask
            mask = mask.at[BoundaryCondition._indices_to_tuple(indices)].set(True)

            return boundary_id, mask

        return set_boundary

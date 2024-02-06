import jax.numpy as jnp
from jax import jit, device_count
import jax.lax as lax
from functools import partial
import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.stream.stream import Stream
from xlb.operator.boundary_condition.boundary_condition import (
    BoundaryCondition,
    ImplementationStep,
)

class HalfwayBounceBack(BoundaryCondition):
    """
    Halfway Bounce-back boundary condition for a lattice Boltzmann method simulation.
    """

    def __init__(
            self,
            set_boundary,
            velocity_set: VelocitySet,
            compute_backend: ComputeBackend = ComputeBackend.JAX,
        ):
        super().__init__(
            set_boundary=set_boundary,
            implementation_step=ImplementationStep.STREAMING,
            velocity_set=velocity_set,
            compute_backend=compute_backend,
        )

    @classmethod
    def from_indices(
            cls,
            indices,
            velocity_set: VelocitySet,
            compute_backend: ComputeBackend = ComputeBackend.JAX,
        ):
        """
        Creates a boundary condition from a list of indices.
        """
       
        # Make stream operator to get edge points
        stream = Stream(velocity_set=velocity_set)

        # Create a mask function
        def set_boundary(ijk, boundary_id, mask, id_number):
            """
            Sets the mask id for the boundary condition.
            Halfway bounce-back is implemented by setting the mask to True for points in the boundary,
            then streaming the mask to get the points on the surface.

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
            local_indices = ijk[tuple(s[:, 0] for s in jnp.split(indices, velocity_set.d, axis=1))]

            # Make mask then stream to get the edge points
            pre_stream_mask = jnp.zeros_like(mask)
            pre_stream_mask = pre_stream_mask.at[tuple([s[:, 0] for s in jnp.split(local_indices, velocity_set.d, axis=1)])].set(True)
            post_stream_mask = stream(pre_stream_mask)

            # Set false for points inside the boundary
            post_stream_mask = post_stream_mask.at[post_stream_mask[..., 0] == True].set(False)

            # Get indices on edges
            edge_indices = jnp.argwhere(post_stream_mask)

            # Set the boundary id
            boundary_id = boundary_id.at[tuple([s[:, 0] for s in jnp.split(local_indices, velocity_set.d, axis=1)])].set(id_number)

            # Set the mask
            mask = mask.at[edge_indices[:, 0], edge_indices[:, 1], edge_indices[:, 2], :].set(post_stream_mask[edge_indices[:, 0], edge_indices[:, 1], edge_indices[:, 2], :])

            return boundary_id, mask

        return cls(
            set_boundary=set_boundary,
            velocity_set=velocity_set,
            compute_backend=compute_backend,
        )


    @partial(jit, static_argnums=(0), donate_argnums=(1, 2, 3, 4))
    def apply_jax(self, f_pre, f_post, boundary, mask):
        flip_mask = boundary[..., jnp.newaxis] & mask
        flipped_f = lax.select(flip_mask, f_pre[..., self.velocity_set.opp_indices], f_post)
        return flipped_f

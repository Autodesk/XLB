# Base class for all equilibriums

from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Tuple

from xlb.global_config import GlobalConfig
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.stream.stream import Stream


class IndicesBoundaryMasker(Operator):
    """
    Operator for creating a boundary mask
    """

    def __init__(
        self,
        stream_indices: bool,
        id_number: int,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend.JAX,
    ):
        super().__init__(velocity_set, precision_policy, compute_backend)

        # Set indices
        self.id_number = id_number
        self.stream_indices = stream_indices

        # Make stream operator
        self.stream = Stream(velocity_set, precision_policy, compute_backend)

    @staticmethod
    def _indices_to_tuple(indices):
        """
        Converts a tensor of indices to a tuple for indexing
        TODO: Might be better to index
        """
        return tuple([indices[:, i] for i in range(indices.shape[1])])

    @Operator.register_backend(ComputeBackend.JAX)
    #@partial(jit, static_argnums=(0), inline=True) TODO: Fix this
    def jax_implementation(self, indicies, start_index, boundary_id, mask, id_number):
        # Get local indices from the meshgrid and the indices
        local_indices = self.indices - np.array(start_index)[np.newaxis, :]

        # Remove any indices that are out of bounds
        local_indices = local_indices[
            (local_indices[:, 0] >= 0)
            & (local_indices[:, 0] < mask.shape[0])
            & (local_indices[:, 1] >= 0)
            & (local_indices[:, 1] < mask.shape[1])
            & (local_indices[:, 2] >= 0)
            & (local_indices[:, 2] < mask.shape[2])
        ]

        # Set the boundary id
        boundary_id = boundary_id.at[self._indices_to_tuple(local_indices)].set(
            id_number
        )

        # Stream mask if necessary
        if self.stream_indices:
            # Make mask then stream to get the edge points
            pre_stream_mask = jnp.zeros_like(mask)
            pre_stream_mask = pre_stream_mask.at[
                self._indices_to_tuple(local_indices)
            ].set(True)
            post_stream_mask = self.stream(pre_stream_mask)

            # Set false for points inside the boundary
            post_stream_mask = post_stream_mask.at[
                post_stream_mask[..., 0] == True
            ].set(False)

            # Get indices on edges
            edge_indices = jnp.argwhere(post_stream_mask)

            # Set the mask
            mask = mask.at[
                edge_indices[:, 0], edge_indices[:, 1], edge_indices[:, 2], :
            ].set(
                post_stream_mask[
                    edge_indices[:, 0], edge_indices[:, 1], edge_indices[:, 2], :
                ]
            )

        else:
            # Set the mask
            mask = mask.at[self._indices_to_tuple(local_indices)].set(True)

        return boundary_id, mask

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, start_index, boundary_id, mask, id_number):
        # Reuse the jax implementation, TODO: implement a warp version
        # Convert to jax
        boundary_id = wp.jax.to_jax(boundary_id)
        mask = wp.jax.to_jax(mask)

        # Call jax implementation
        boundary_id, mask = self.jax_implementation(
            start_index, boundary_id, mask, id_number
        )

        # Convert back to warp
        boundary_id = wp.jax.to_warp(boundary_id)
        mask = wp.jax.to_warp(mask)

        return boundary_id, mask

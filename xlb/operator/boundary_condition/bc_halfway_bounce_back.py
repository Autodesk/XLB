"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit
import jax.lax as lax
from functools import partial
import warp as wp
from typing import Any

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.boundary_condition.boundary_condition import (
    ImplementationStep,
    BoundaryCondition,
)
from xlb.operator.boundary_condition.boundary_condition_registry import (
    boundary_condition_registry,
)


class HalfwayBounceBackBC(BoundaryCondition):
    """
    Halfway Bounce-back boundary condition for a lattice Boltzmann method simulation.

    TODO: Implement moving boundary conditions for this
    """

    id = boundary_condition_registry.register_boundary_condition(__qualname__)

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
    ):
        # Call the parent constructor
        super().__init__(
            ImplementationStep.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
            mesh_vertices,
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def apply_jax(self, f_pre, f_post, boundary_map, missing_mask):
        boundary = boundary_map == self.id
        new_shape = (self.velocity_set.q,) + boundary.shape[1:]
        boundary = lax.broadcast_in_dim(boundary, new_shape, tuple(range(self.velocity_set.d + 1)))
        return jnp.where(
            jnp.logical_and(missing_mask, boundary),
            f_pre[self.velocity_set.opp_indices],
            f_post,
        )

    def _construct_warp(self):
        # Set local constants
        _opp_indices = self.velocity_set.wp_opp_indices

        # Construct the functional for this BC
        @wp.func
        def functional(
            f_pre: Any,
            f_post: Any,
            f_aux: Any,
            missing_mask: Any,
        ):
            # Post-streaming values are only modified at missing direction
            _f = f_post
            for l in range(self.velocity_set.q):
                # If the mask is missing then take the opposite index
                if missing_mask[l] == wp.uint8(1):
                    # Get the pre-streaming distribution function in oppisite direction
                    _f[l] = f_pre[_opp_indices[l]]

            return _f

        # Construct the warp kernel
        @wp.kernel
        def kernel2d(
            f_pre: wp.array3d(dtype=Any),
            f_post: wp.array3d(dtype=Any),
            boundary_map: wp.array3d(dtype=wp.uint8),
            missing_mask: wp.array3d(dtype=wp.bool),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            # read tid data
            _f_pre, _f_post, _boundary_id, _missing_mask = self._get_thread_data_2d(f_pre, f_post, boundary_map, missing_mask, index)

            # Apply the boundary condition
            if _boundary_id == wp.uint8(HalfwayBounceBackBC.id):
                _f_aux = _f_post
                _f = functional(_f_pre, _f_post, _f_aux, _missing_mask)
            else:
                _f = _f_post

            # Write the distribution function
            for l in range(self.velocity_set.q):
                f_post[l, index[0], index[1]] = _f[l]

        # Construct the warp kernel
        @wp.kernel
        def kernel3d(
            f_pre: wp.array4d(dtype=Any),
            f_post: wp.array4d(dtype=Any),
            boundary_map: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # read tid data
            _f_pre, _f_post, _boundary_id, _missing_mask = self._get_thread_data_3d(f_pre, f_post, boundary_map, missing_mask, index)

            # Apply the boundary condition
            if _boundary_id == wp.uint8(HalfwayBounceBackBC.id):
                _f_aux = _f_post
                _f = functional(_f_pre, _f_post, _f_aux, _missing_mask)
            else:
                _f = _f_post

            # Write the distribution function
            for l in range(self.velocity_set.q):
                f_post[l, index[0], index[1], index[2]] = _f[l]

        kernel = kernel3d if self.velocity_set.d == 3 else kernel2d

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, boundary_map, missing_mask):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, boundary_map, missing_mask],
            dim=f_pre.shape[1:],
        )
        return f_post

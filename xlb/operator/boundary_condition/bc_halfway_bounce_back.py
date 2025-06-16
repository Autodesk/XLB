"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit
import jax.lax as lax
from functools import partial
import warp as wp
from typing import Any, Union, Tuple, Callable
import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.boundary_condition.boundary_condition import (
    ImplementationStep,
    BoundaryCondition,
    HelperFunctionsBC,
)
from xlb.operator.boundary_masker.mesh_voxelization_method import MeshVoxelizationMethod


class HalfwayBounceBackBC(BoundaryCondition):
    """
    Halfway Bounce-back boundary condition for a lattice Boltzmann method simulation.

    TODO: Implement moving boundary conditions for this
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
        voxelization_method: MeshVoxelizationMethod = None,
        profile: Callable = None,
        prescribed_value: Union[float, Tuple[float, ...], np.ndarray] = None,
    ):
        # Call the parent constructor
        super().__init__(
            ImplementationStep.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
            mesh_vertices,
            voxelization_method,
        )

        # This BC needs padding for finding missing directions when imposed on a geometry that is in the domain interior
        self.needs_padding = True

        # This BC class accepts both constant prescribed values of velocity with keyword "prescribed_value" or
        # velocity profiles given by keyword "profile" which must be a callable function.
        self.profile = profile

        # A flag to enable moving wall treatment when either "prescribed_value" or "profile" are provided.
        self.needs_moving_wall_treatment = False

        if (profile is not None) or (prescribed_value is not None):
            self.needs_moving_wall_treatment = True

        # Handle no-slip BCs if neither prescribed_value or profile are provided.
        if prescribed_value is None and profile is None:
            print(f"WARNING! Assuming no-slip condition for BC type = {self.__class__.__name__}!")
            prescribed_value = [0] * self.velocity_set.d

        # Handle prescribed value if provided
        if prescribed_value is not None:
            if profile is not None:
                raise ValueError("Cannot specify both profile and prescribed_value")

            # Ensure prescribed_value is a NumPy array of floats
            if isinstance(prescribed_value, (tuple, list, np.ndarray)):
                prescribed_value = np.asarray(prescribed_value, dtype=np.float64)
            else:
                raise ValueError("Velocity prescribed_value must be a tuple, list, or array")

            # Handle 2D velocity sets
            if self.velocity_set.d == 2:
                assert len(prescribed_value) == 2, "For 2D velocity set, prescribed_value must be a tuple or array of length 2!"
                prescribed_value = np.array([prescribed_value[0], prescribed_value[1], 0.0], dtype=np.float64)

            # create a constant prescribed profile
            prescribed_value = wp.vec(3, dtype=self.compute_dtype)(prescribed_value)

            @wp.func
            def prescribed_profile_warp(index: wp.vec3i, time: Any):
                return wp.vec3(prescribed_value[0], prescribed_value[1], prescribed_value[2])

            self.profile = prescribed_profile_warp

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        boundary = bc_mask == self.id
        new_shape = (self.velocity_set.q,) + boundary.shape[1:]
        boundary = lax.broadcast_in_dim(boundary, new_shape, tuple(range(self.velocity_set.d + 1)))
        return jnp.where(
            jnp.logical_and(missing_mask, boundary),
            f_pre[self.velocity_set.opp_indices],
            f_post,
        )

    def _construct_warp(self):
        # load helper functions
        bc_helper = HelperFunctionsBC(velocity_set=self.velocity_set, precision_policy=self.precision_policy, compute_backend=self.compute_backend)

        # Set local constants
        _opp_indices = self.velocity_set.opp_indices

        # Construct the functional for this BC
        @wp.func
        def functional(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
        ):
            # Get wall velocity
            u_wall = self.profile(index, timestep)

            # Post-streaming values are only modified at missing direction
            _f = f_post
            for l in range(self.velocity_set.q):
                # If the mask is missing then take the opposite index
                if missing_mask[l] == wp.uint8(1):
                    # Get the pre-streaming distribution function in oppisite direction
                    _f[l] = f_pre[_opp_indices[l]]

                    # Add contribution due to moving_wall to f_missing
                    if wp.static(self.needs_moving_wall_treatment):
                        _f[l] += bc_helper.moving_wall_fpop_correction(u_wall, l)

            return _f

        kernel = self._construct_kernel(functional)

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, bc_mask, missing_mask],
            dim=f_pre.shape[1:],
        )
        return f_post

"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit
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
from xlb.operator.equilibrium import QuadraticEquilibrium


class ZouHeBC(BoundaryCondition):
    """
    Zou-He boundary condition for a lattice Boltzmann method simulation.

    This class implements the Zou-He boundary condition, which is a non-equilibrium bounce-back boundary condition.
    It can be used to set inflow and outflow boundary conditions with prescribed pressure or velocity.
    """

    id = boundary_condition_registry.register_boundary_condition(__qualname__)

    def __init__(
        self,
        bc_type=None,
        prescribed_value=None,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
    ):
        assert bc_type in ["velocity", "pressure"], f'The boundary type must be either "velocity" or "pressure"'
        self.bc_type = bc_type
        self.equilibrium_operator = QuadraticEquilibrium()

        # Call the parent constructor
        super().__init__(
            ImplementationStep.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
        )

        # Set the prescribed value for pressure or velocity
        dim = self.velocity_set.d
        self.prescribed_value = jnp.array(prescribed_value)[:, None, None, None] if dim == 3 else jnp.array(prescribed_value)[:, None, None]
        # TODO: this won't work if the prescribed values are a profile with the length of bdry indices!

    @partial(jit, static_argnums=(0,), inline=True)
    def _get_known_middle_mask(self, missing_mask):
        known_mask = missing_mask[self.velocity_set.opp_indices]
        middle_mask = ~(missing_mask | known_mask)
        return known_mask, middle_mask

    @partial(jit, static_argnums=(0,), inline=True)
    def _get_normal_vec(self, missing_mask):
        main_c = self.velocity_set.c[:, self.velocity_set.main_indices]
        m = missing_mask[self.velocity_set.main_indices]
        normals = -jnp.tensordot(main_c, m, axes=(-1, 0))
        return normals

    @partial(jit, static_argnums=(0,), inline=True)
    def get_rho(self, fpop, missing_mask):
        if self.bc_type == "velocity":
            vel = self.get_vel(fpop, missing_mask)
            rho = self.calculate_rho(fpop, vel, missing_mask)
        elif self.bc_type == "pressure":
            rho = self.prescribed_value
        else:
            raise ValueError(f"type = {self.bc_type} not supported! Use 'pressure' or 'velocity'.")
        return rho

    @partial(jit, static_argnums=(0,), inline=True)
    def get_vel(self, fpop, missing_mask):
        if self.bc_type == "velocity":
            vel = self.prescribed_value
        elif self.bc_type == "pressure":
            rho = self.get_rho(fpop, missing_mask)
            vel = self.calculate_vel(fpop, rho, missing_mask)
        else:
            raise ValueError(f"type = {self.bc_type} not supported! Use 'pressure' or 'velocity'.")
        return vel

    @partial(jit, static_argnums=(0,), inline=True)
    def calculate_vel(self, fpop, rho, missing_mask):
        """
        Calculate velocity based on the prescribed pressure/density (Zou/He BC)
        """

        normals = self._get_normal_vec(missing_mask)
        known_mask, middle_mask = self._get_known_middle_mask(missing_mask)

        unormal = -1.0 + 1.0 / rho * (jnp.sum(fpop * middle_mask, axis=-1, keepdims=True) + 2.0 * jnp.sum(fpop * known_mask, axis=-1, keepdims=True))

        # Return the above unormal as a normal vector which sets the tangential velocities to zero
        vel = unormal * normals
        return vel

    @partial(jit, static_argnums=(0,), inline=True)
    def calculate_rho(self, fpop, vel, missing_mask):
        """
        Calculate density based on the prescribed velocity (Zou/He BC)
        """
        normals = self._get_normal_vec(missing_mask)
        known_mask, middle_mask = self._get_known_middle_mask(missing_mask)
        unormal = jnp.sum(normals * vel, keepdims=True, axis=0)
        rho = (1.0 / (1.0 + unormal)) * (jnp.sum(fpop * middle_mask, axis=0, keepdims=True) + 2.0 * jnp.sum(fpop * known_mask, axis=0, keepdims=True))
        return rho

    @partial(jit, static_argnums=(0,), inline=True)
    def calculate_equilibrium(self, fpop, missing_mask):
        """
        This is the ZouHe method of calculating the missing macroscopic variables at the boundary.
        """
        rho = self.get_rho(fpop, missing_mask)
        vel = self.get_vel(fpop, missing_mask)

        # compute feq at the boundary
        feq = self.equilibrium_operator(rho, vel)
        return feq

    @partial(jit, static_argnums=(0,), inline=True)
    def bounceback_nonequilibrium(self, fpop, feq, missing_mask):
        """
        Calculate unknown populations using bounce-back of non-equilibrium populations
        a la original Zou & He formulation
        """
        opp = self.velocity_set.opp_indices
        fknown = fpop[opp] + feq - feq[opp]
        fpop = jnp.where(missing_mask, fknown, fpop)
        return fpop

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def apply_jax(self, f_pre, f_post, boundary_mask, missing_mask):
        # creat a mask to slice boundary cells
        boundary = boundary_mask == self.id
        boundary = jnp.repeat(boundary, self.velocity_set.q, axis=0)

        # compute the equilibrium based on prescribed values and the type of BC
        feq = self.calculate_equilibrium(f_post, missing_mask)

        # set the unknown f populations based on the non-equilibrium bounce-back method
        f_post_bd = self.bounceback_nonequilibrium(f_post, feq, missing_mask)
        f_post = jnp.where(boundary, f_post_bd, f_post)
        return f_post

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _c = self.velocity_set.wp_c
        _opp_indices = self.velocity_set.wp_opp_indices
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)  # TODO fix vec bool

        @wp.func
        def functional(
            f_pre: Any,
            f_post: Any,
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
            boundary_mask: wp.array3d(dtype=wp.uint8),
            missing_mask: wp.array3d(dtype=wp.bool),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec3i(i, j)

            # read tid data
            _f_pre, _f_post, _boundary_id, _missing_mask = self._get_thread_data_2d(f_pre, f_post, boundary_mask, missing_mask, index)

            # Apply the boundary condition
            if _boundary_id == wp.uint8(ZouHeBC.id):
                _f = functional(_f_pre, _f_post, _missing_mask)
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
            boundary_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # read tid data
            _f_pre, _f_post, _boundary_id, _missing_mask = self._get_thread_data_3d(f_pre, f_post, boundary_mask, missing_mask, index)

            # Apply the boundary condition
            if _boundary_id == wp.uint8(ZouHeBC.id):
                _f = functional(_f_pre, _f_post, _missing_mask)
            else:
                _f = _f_post

            # Write the distribution function
            for l in range(self.velocity_set.q):
                f_post[l, index[0], index[1], index[2]] = _f[l]

        kernel = kernel3d if self.velocity_set.d == 3 else kernel2d

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, boundary_mask, missing_mask):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, boundary_mask, missing_mask],
            dim=f_pre.shape[1:],
        )
        return f_post

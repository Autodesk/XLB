"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit
import jax.lax as lax
from functools import partial
import warp as wp
from typing import Any, Union, Tuple
import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.boundary_condition.boundary_condition import (
    ImplementationStep,
    BoundaryCondition,
)
from xlb.operator.equilibrium import QuadraticEquilibrium
import jax


class ZouHeBC(BoundaryCondition):
    """
    Zou-He boundary condition for a lattice Boltzmann method simulation.

    This method applies the Zou-He boundary condition by first computing the equilibrium distribution functions based
    on the prescribed values and the type of boundary condition, and then setting the unknown distribution functions
    based on the non-equilibrium bounce-back method.
    Tangential velocity is not ensured to be zero by adding transverse contributions based on
    Hecth & Harting (2010) (doi:10.1088/1742-5468/2010/01/P01018) as it caused numerical instabilities at higher
    Reynolds numbers. One needs to use "Regularized" BC at higher Reynolds.
    """

    def __init__(
        self,
        bc_type,
        profile=None,
        prescribed_value: Union[float, Tuple[float, ...], np.ndarray] = None,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
    ):
        # Important Note: it is critical to add id inside __init__ for this BC because different instantiations of this BC
        # may have different types (velocity or pressure).
        assert bc_type in ["velocity", "pressure"], f"type = {bc_type} not supported! Use 'pressure' or 'velocity'."
        self.bc_type = bc_type
        self.equilibrium_operator = QuadraticEquilibrium()
        self.profile = profile

        # Call the parent constructor
        super().__init__(
            ImplementationStep.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
            mesh_vertices,
        )

        # Handle prescribed value if provided
        if prescribed_value is not None:
            if profile is not None:
                raise ValueError("Cannot specify both profile and prescribed_value")

            # Convert input to numpy array for validation
            if isinstance(prescribed_value, (tuple, list)):
                prescribed_value = np.array(prescribed_value, dtype=np.float64)
            elif isinstance(prescribed_value, (int, float)):
                if bc_type == "pressure":
                    prescribed_value = float(prescribed_value)
                else:
                    raise ValueError("Velocity prescribed_value must be a tuple or array")
            elif isinstance(prescribed_value, np.ndarray):
                prescribed_value = prescribed_value.astype(np.float64)

            # Validate prescribed value
            if bc_type == "velocity":
                if not isinstance(prescribed_value, np.ndarray):
                    raise ValueError("Velocity prescribed_value must be an array-like")

                # Check for non-zero elements - only one element should be non-zero
                non_zero_count = np.count_nonzero(prescribed_value)
                if non_zero_count > 1:
                    raise ValueError("This BC only supports normal prescribed values (only one non-zero element allowed)")

            self.prescribed_value = prescribed_value
            self.profile = self._create_constant_prescribed_profile()

        # This BC needs auxilary data initialization before streaming
        self.needs_aux_init = True

        # This BC needs auxilary data recovery after streaming
        self.needs_aux_recovery = True

        # This BC needs one auxilary data for the density or normal velocity
        self.num_of_aux_data = 1

        # This BC needs padding for finding missing directions when imposed on a geometry that is in the domain interior
        self.needs_padding = True

    def _create_constant_prescribed_profile(self):
        if self.bc_type == "velocity":

            @wp.func
            def prescribed_profile_warp(index: wp.vec3i):
                # Get the non-zero value from prescribed_value
                value = wp.static(
                    self.precision_policy.store_precision.wp_dtype(float(self.prescribed_value[np.nonzero(self.prescribed_value)[0][0]]))
                )
                return wp.vec(value, length=1)

            def prescribed_profile_jax():
                return jnp.array(self.prescribed_value, dtype=self.precision_policy.store_precision.jax_dtype).reshape(-1, 1)

        else:  # pressure

            @wp.func
            def prescribed_profile_warp(index: wp.vec3i):
                value = wp.static(self.precision_policy.store_precision.wp_dtype(self.prescribed_value))
                return wp.vec(value, length=1)

            def prescribed_profile_jax():
                return jnp.array(self.prescribed_value)

        if self.compute_backend == ComputeBackend.JAX:
            return prescribed_profile_jax
        elif self.compute_backend == ComputeBackend.WARP:
            return prescribed_profile_warp

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

    @partial(jit, static_argnums=(0, 2, 3), inline=True)
    def _broadcast_prescribed_values(self, prescribed_values, prescribed_values_shape, target_shape):
        """
        Broadcasts `prescribed_values` to `target_shape` following specific rules:

        - If `prescribed_values_shape` is (2, 1) or (3, 1) (for constant profiles),
          broadcast along the last 2 or 3 dimensions of `target_shape` respectively.
        - For other shapes, identify mismatched dimensions and broadcast only in that direction.
        """
        # Determine the number of dimensions to match
        num_dims_prescribed = len(prescribed_values_shape)
        num_dims_target = len(target_shape)

        if num_dims_prescribed > num_dims_target:
            raise ValueError("prescribed_values has more dimensions than target_shape")

        # Insert singleton dimensions after the first dimension to match target_shape
        if num_dims_prescribed < num_dims_target:
            # Number of singleton dimensions to add
            num_singleton = num_dims_target - num_dims_prescribed

            if num_dims_prescribed == 0:
                # If prescribed_values is scalar, reshape to all singleton dimensions
                prescribed_values_shape = (1,) * num_dims_target
            else:
                # Insert singleton dimensions after the first dimension
                prescribed_values_shape = (prescribed_values_shape[0], *(1,) * num_singleton, *prescribed_values_shape[1:])
                prescribed_values = prescribed_values.reshape(prescribed_values_shape)

        # Create broadcast shape based on the rules
        broadcast_shape = []
        for pv_dim, tgt_dim in zip(prescribed_values_shape, target_shape):
            if pv_dim == 1 or pv_dim == tgt_dim:
                broadcast_shape.append(tgt_dim)
            else:
                raise ValueError(f"Cannot broadcast dimension {pv_dim} to {tgt_dim}")

        return jnp.broadcast_to(prescribed_values, target_shape)

    @partial(jit, static_argnums=(0,), inline=True)
    def get_rho(self, fpop, missing_mask):
        if self.bc_type == "velocity":
            target_shape = (self.velocity_set.d,) + fpop.shape[1:]
            vel = self._broadcast_prescribed_values(self.prescribed_values, self.prescribed_values.shape, target_shape)
            rho = self.calculate_rho(fpop, vel, missing_mask)
        elif self.bc_type == "pressure":
            rho = self.prescribed_values
        else:
            raise ValueError(f"type = {self.bc_type} not supported! Use 'pressure' or 'velocity'.")
        return rho

    @partial(jit, static_argnums=(0,), inline=True)
    def get_vel(self, fpop, missing_mask):
        if self.bc_type == "velocity":
            target_shape = (self.velocity_set.d,) + fpop.shape[1:]
            vel = self._broadcast_prescribed_values(self.prescribed_values, self.prescribed_values.shape, target_shape)
        elif self.bc_type == "pressure":
            rho = self.prescribed_values
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
        fsum = jnp.sum(fpop * middle_mask, axis=0, keepdims=True) + 2.0 * jnp.sum(fpop * known_mask, axis=0, keepdims=True)
        unormal = -1.0 + fsum / rho

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
        fsum = jnp.sum(fpop * middle_mask, axis=0, keepdims=True) + 2.0 * jnp.sum(fpop * known_mask, axis=0, keepdims=True)
        rho = fsum / (1.0 + unormal)
        return rho

    @partial(jit, static_argnums=(0,), inline=True)
    def calculate_equilibrium(self, f_post, missing_mask):
        """
        This is the ZouHe method of calculating the missing macroscopic variables at the boundary.
        """
        rho = self.get_rho(f_post, missing_mask)
        vel = self.get_vel(f_post, missing_mask)

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
    def jax_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        # creat a mask to slice boundary cells
        boundary = bc_mask == self.id
        new_shape = (self.velocity_set.q,) + boundary.shape[1:]
        boundary = lax.broadcast_in_dim(boundary, new_shape, tuple(range(self.velocity_set.d + 1)))

        # compute the equilibrium based on prescribed values and the type of BC
        feq = self.calculate_equilibrium(f_post, missing_mask)

        # set the unknown f populations based on the non-equilibrium bounce-back method
        f_post_bd = self.bounceback_nonequilibrium(f_post, feq, missing_mask)
        f_post = jnp.where(boundary, f_post_bd, f_post)
        return f_post

    def _construct_warp(self):
        # load helper functions
        from xlb.helper.bc_warp_functions import get_normal_vectors, get_bc_fsum, bounceback_nonequilibrium

        # Set local constants
        _d = self.velocity_set.d
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices

        @wp.func
        def functional_velocity(
            index: Any,
            timestep: Any,
            _missing_mask: Any,
            f_pre: Any,
            f_post: Any,
            _f_pre: Any,
            _f_post: Any,
        ):
            # Post-streaming values are only modified at missing direction
            _f = _f_post

            # Find normal vector
            normals = get_normal_vectors(_missing_mask)

            # calculate rho
            fsum = get_bc_fsum(_f, _missing_mask)
            unormal = self.compute_dtype(0.0)

            # Find the value of u from the missing directions
            for l in range(_q):
                # Since we are only considering normal velocity, we only need to find one value (all values are the same in the missing directions)
                if _missing_mask[l] == wp.uint8(1):
                    # Create velocity vector by multiplying the prescribed value with the normal vector
                    # TODO: This can be optimized by saving _missing_mask[l] in the bc class later since it is the same for all boundary cells
                    prescribed_value = f_post[_opp_indices[l], index[0], index[1], index[2]]
                    _u = -prescribed_value * normals
                    break

            for d in range(_d):
                unormal += _u[d] * normals[d]

            _rho = fsum / (self.compute_dtype(1.0) + unormal)

            # impose non-equilibrium bounceback
            _feq = self.equilibrium_operator.warp_functional(_rho, _u)
            _f = bounceback_nonequilibrium(_f, _feq, _missing_mask)
            return _f

        @wp.func
        def functional_pressure(
            index: Any,
            timestep: Any,
            _missing_mask: Any,
            f_pre: Any,
            f_post: Any,
            _f_pre: Any,
            _f_post: Any,
        ):
            # Post-streaming values are only modified at missing direction
            _f = _f_post

            # Find normal vector
            normals = get_normal_vectors(_missing_mask)

            # Find the value of rho from the missing directions
            for q in range(_q):
                # Since we need only one scalar value, we only need to find one value (all values are the same in the missing directions)
                if _missing_mask[q] == wp.uint8(1):
                    _rho = f_post[_opp_indices[q], index[0], index[1], index[2]]
                    break

            # calculate velocity
            fsum = get_bc_fsum(_f, _missing_mask)
            unormal = -self.compute_dtype(1.0) + fsum / _rho
            _u = unormal * normals

            # impose non-equilibrium bounceback
            feq = self.equilibrium_operator.warp_functional(_rho, _u)
            _f = bounceback_nonequilibrium(_f, feq, _missing_mask)
            return _f

        if self.bc_type == "velocity":
            functional = functional_velocity
        elif self.bc_type == "pressure":
            functional = functional_pressure

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

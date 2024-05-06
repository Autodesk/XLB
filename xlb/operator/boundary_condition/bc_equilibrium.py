"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit, device_count
import jax.lax as lax
from functools import partial
import numpy as np
import warp as wp
from typing import Tuple, Any

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


class EquilibriumBC(BoundaryCondition):
    """
    Full Bounce-back boundary condition for a lattice Boltzmann method simulation.
    """

    id = boundary_condition_registry.register_boundary_condition(__qualname__)

    def __init__(
        self,
        rho: float,
        u: Tuple[float, float, float],
        equilibrium_operator: Operator,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        # Store the equilibrium information
        self.rho = rho
        self.u = u
        self.equilibrium_operator = equilibrium_operator

        # Call the parent constructor
        super().__init__(
            ImplementationStep.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, boundary_id_field, missing_mask):
        feq = self.equilibrium_operator(jnp.array([self.rho]), jnp.array(self.u))
        new_shape = feq.shape + (1,) * self.velocity_set.d
        feq = lax.broadcast_in_dim(feq, new_shape, [0])
        boundary = boundary_id_field == self.id

        return jnp.where(boundary, feq, f_post)

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _rho = wp.float32(self.rho)
        _u = _u_vec(self.u[0], self.u[1], self.u[2]) if self.velocity_set.d == 3 else _u_vec(self.u[0], self.u[1])
        _missing_mask_vec = wp.vec(
            self.velocity_set.q, dtype=wp.uint8
        )  # TODO fix vec bool

        # Construct the funcional to get streamed indices
        @wp.func
        def functional2d():
            _f = self.equilibrium_operator.warp_functional(_rho, _u)
            return _f

        # Construct the warp kernel
        @wp.kernel
        def kernel2d(
            f_pre: wp.array3d(dtype=Any),
            f_post: wp.array3d(dtype=Any),
            boundary_id_field: wp.array3d(dtype=wp.uint8),
            missing_mask: wp.array3d(dtype=wp.bool),
            f: wp.array3d(dtype=Any),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            # Get the boundary id and missing mask
            _boundary_id = boundary_id_field[0, index[0], index[1]]
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # TODO fix vec bool
                if missing_mask[l, index[0], index[1]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)

            # Apply the boundary condition
            if _boundary_id == wp.uint8(EquilibriumBC.id):
                _f = functional2d()
            else:
                _f = _f_vec()
                for l in range(self.velocity_set.q):
                    _f[l] = f_post[l, index[0], index[1]]

            # Write the result
            for l in range(self.velocity_set.q):
                f[l, index[0], index[1]] = _f[l]

        @wp.func
        def functional3d():
            _f = self.equilibrium_operator.warp_functional(_rho, _u)
            return _f

        # Construct the warp kernel
        @wp.kernel
        def kernel3d(
            f_pre: wp.array4d(dtype=Any),
            f_post: wp.array4d(dtype=Any),
            boundary_id_field: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
            f: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Get the boundary id and missing mask
            _boundary_id = boundary_id_field[0, index[0], index[1], index[2]]
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # TODO fix vec bool
                if missing_mask[l, index[0], index[1], index[2]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)

            # Apply the boundary condition
            if _boundary_id == wp.uint8(EquilibriumBC.id):
                _f = functional3d()
            else:
                _f = _f_vec()
                for l in range(self.velocity_set.q):
                    _f[l] = f_post[l, index[0], index[1], index[2]]

            # Write the result
            for l in range(self.velocity_set.q):
                f[l, index[0], index[1], index[2]] = _f[l]

        kernel = kernel3d if self.velocity_set.d == 3 else kernel2d
        functional = functional3d if self.velocity_set.d == 3 else functional2d

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, boundary_id_field, missing_mask, f):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, boundary_id_field, missing_mask, f],
            dim=f_pre.shape[1:],
        )
        return f
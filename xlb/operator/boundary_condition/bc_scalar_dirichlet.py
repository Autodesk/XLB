"""
Dirichlet boundary condition for a scalar field (anti-bounce-back).

Prescribes the value of a transported scalar (e.g. temperature) on a wall located halfway
between the boundary node and the adjacent solid/exterior node.
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
from xlb.operator.boundary_masker.mesh_voxelization_method import MeshVoxelizationMethod


class ScalarDirichletBC(BoundaryCondition):
    """
    Anti-bounce-back boundary condition prescribing the value of a transported scalar.

    The missing populations are reconstructed as::

        g[l] = -g_pre[opp[l]] + 2 * w[l] * value

    which imposes ``phi = value`` halfway between the boundary node and its solid neighbour.
    The scheme is consistent with the linear scalar equilibrium of
    :class:`~xlb.operator.equilibrium.LinearEquilibrium`, because for that equilibrium
    ``g_eq[l] + g_eq[opp[l]] = 2 * w[l] * phi`` regardless of the advecting velocity, and it
    places the wall exactly halfway for a source-free problem.

    Note
    ----
    In the presence of a volumetric source the wall treatment carries a residual defect of
    order *source*, because the incoming and outgoing populations of a wall link are
    produced half a time step apart while the source is applied per full step. Since the
    lattice source scales as ``dx^2`` under diffusive scaling, the resulting error is second
    order in ``dx``, in line with the rest of the scheme.

    Parameters
    ----------
    value : float
        Prescribed scalar value on the wall (e.g. wall temperature in lattice units).
    """

    def __init__(
        self,
        value: float = 0.0,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
        voxelization_method: MeshVoxelizationMethod = None,
    ):
        # Store the prescribed wall value
        self.value = value

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

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        boundary = bc_mask == self.id
        new_shape = (self.velocity_set.q,) + boundary.shape[1:]
        boundary = lax.broadcast_in_dim(boundary, new_shape, tuple(range(self.velocity_set.d + 1)))

        # Anti-bounce-back on the missing directions
        w = self.velocity_set.w.reshape((-1,) + (1,) * self.velocity_set.d)
        f_wall = -f_pre[self.velocity_set.opp_indices] + 2.0 * w * self.compute_dtype(self.value)

        return jnp.where(jnp.logical_and(missing_mask, boundary), f_wall, f_post)

    def _construct_warp(self):
        # Set local constants
        _opp_indices = self.velocity_set.opp_indices
        _w = self.velocity_set.w
        _value = self.compute_dtype(self.value)

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
            # Post-streaming values are only modified at missing directions
            _f = f_post
            for l in range(self.velocity_set.q):
                if missing_mask[l] == wp.uint8(1):
                    _f[l] = -f_pre[_opp_indices[l]] + self.compute_dtype(2.0) * _w[l] * _value

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

    def _construct_neon(self):
        functional, _ = self._construct_warp()
        return functional, None

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        # raise exception as this feature is not implemented yet
        raise NotImplementedError("This feature is not implemented in XLB with the NEON backend yet.")

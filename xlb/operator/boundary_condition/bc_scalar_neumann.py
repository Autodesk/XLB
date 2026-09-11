"""
Neumann boundary condition for a scalar field (bounce-back with flux injection).

Prescribes the normal diffusive flux of a transported scalar (e.g. an imposed heat flux or
an adiabatic wall) on a wall located halfway between the boundary node and the adjacent
solid/exterior node.
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


class ScalarNeumannBC(BoundaryCondition):
    """
    Bounce-back boundary condition prescribing the normal diffusive flux of a scalar.

    The missing populations are reconstructed as::

        g[l] = g_pre[opp[l]] + 2 * w[l] * flux / cs^2

    Plain bounce-back (``flux = 0``) conserves the scalar at the wall and therefore imposes
    a zero normal gradient, i.e. an adiabatic wall. The extra term injects the prescribed
    amount of scalar per node and per time step: for an axis-aligned wall
    ``sum_missing 2 * w[l] / cs^2 = 1``, so the total injection is exactly ``flux``.

    The relation between *flux* and the physical wall heat flux ``q_wall`` of
    ``rho * cp * dT/dt = k * laplacian(T)`` is

        flux = alpha_lattice * dT/dn|_wall = q_wall * dt / (rho * cp * dx)

    with ``n`` the outward wall normal, so that a positive *flux* heats the domain.

    Warning
    -------
    **Do not use this condition on an outlet or any boundary the fluid crosses.** It is a
    wall condition and assumes the fluid does not move through the boundary.

    Bounce-back nulls the antisymmetric part of the closure relation, which imposes zero
    **total** normal flux, advective plus diffusive. On a no-slip wall the advective part
    is already zero, so this correctly gives an adiabatic wall. On an open boundary the
    normal is the direction the fluid crosses, ``c[l] . u != 0`` on the missing link, and
    the outgoing scalar is reflected straight back in. Relative to the equilibrium
    ``g_eq[l] = w[l] * phi * (1 + c[l] . u / cs^2)`` the reflected population is wrong by

        2 * w[l] * phi * (c[l] . u) / cs^2

    per node per step, which acts as a source sitting on the outlet and drives the scalar
    out of its physical range within a few thousand steps. Use
    :class:`~xlb.operator.boundary_condition.ScalarOutflowBC` for open boundaries.

    Note
    ----
    The identity ``sum_missing 2 * w[l] / cs^2 = 1`` holds for flat, axis-aligned walls.
    On concave corners and on voxelized curved geometry the injected amount deviates from
    the prescribed flux, so a non-zero *flux* should only be used on flat walls. The
    adiabatic case (``flux = 0``) is unaffected.

    Parameters
    ----------
    flux : float
        Prescribed normal flux in lattice units, positive when the scalar enters the domain.
        Defaults to ``0.0``, i.e. an adiabatic / zero-gradient wall.
    """

    def __init__(
        self,
        flux: float = 0.0,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
        voxelization_method: MeshVoxelizationMethod = None,
    ):
        # Store the prescribed wall flux
        self.flux = flux

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

        # Bounce-back on the missing directions, plus the prescribed flux injection
        w = self.velocity_set.w.reshape((-1,) + (1,) * self.velocity_set.d)
        flux_component = 2.0 * self.velocity_set.inv_cs2 * w * self.compute_dtype(self.flux)
        f_wall = f_pre[self.velocity_set.opp_indices] + flux_component

        return jnp.where(jnp.logical_and(missing_mask, boundary), f_wall, f_post)

    def _construct_warp(self):
        # Set local constants
        _opp_indices = self.velocity_set.opp_indices
        _w = self.velocity_set.w
        _inv_cs2 = self.velocity_set.inv_cs2
        _flux = self.compute_dtype(self.flux)

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
                    _f[l] = f_pre[_opp_indices[l]]

                    if wp.static(self.flux != 0.0):
                        _f[l] += self.compute_dtype(2.0) * _inv_cs2 * _w[l] * _flux

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

"""
Open (outflow) boundary condition for a scalar field.

Lets a transported scalar leave the domain through a boundary that the fluid crosses,
without imposing a value on it and without reflecting it back into the domain.
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


class ScalarOutflowBC(BoundaryCondition):
    """
    Open boundary for a transported scalar, for use where the fluid leaves the domain.

    The missing populations are filled by zeroth-order extrapolation along the boundary
    normal, i.e. the node keeps the population it held before streaming::

        g[l] = g_pre[l]        on the missing directions

    which is the zeroth-order population-extrapolation outflow treatment analysed by
    Yang (2013). It imposes no value and adds no scalar, so a plume simply convects out.

    Why not bounce-back
    -------------------
    It is tempting to use :class:`ScalarNeumannBC` with ``flux = 0`` at an outlet, on the
    grounds that an outlet has zero normal gradient. That is wrong, and badly so.

    Bounce-back sets the *antisymmetric* part of the closure relation to zero, which
    imposes zero **total** normal flux -- advective plus diffusive. At a solid wall the
    fluid does not move through the boundary, the advective part is already zero, and
    bounce-back correctly imposes zero diffusive flux. At an outlet the boundary normal
    is precisely the direction the fluid crosses, so ``c[l] . u != 0`` on the missing
    link, and bounce-back reflects the outgoing scalar back into the domain. Writing the
    equilibrium ``g_eq[l] = w[l] * phi * (1 + c[l] . u / cs^2)``, the reflected population
    is short by

        2 * w[l] * phi * (c[l] . u) / cs^2

    every node and every step. This is a source term sitting on the outlet, not a
    zero-gradient condition, and it drives the solution out of the physical range of the
    scalar within a few thousand steps.

    The rule of thumb is Ginzburg's symmetric/antisymmetric split:

    ==============================  ====================  =========================
    Boundary                        Correct scheme        XLB class
    ==============================  ====================  =========================
    Wall, ``u = 0``, zero flux      bounce-back           :class:`ScalarNeumannBC`
    Wall, prescribed value          anti-bounce-back      :class:`ScalarDirichletBC`
    Open boundary, ``u . n != 0``   extrapolation         this class
    ==============================  ====================  =========================

    Convective variant
    ------------------
    Setting *u_conv* to a positive value switches to the convective (Sommerfeld) outflow
    of Yang (2013), which solves ``d(g)/dt + U d(g)/dn = 0`` along the boundary normal::

        g[l] = (g_pre[l] + U * g_neighbour[l]) / (1 + U)

    with ``U = u_conv`` the convection speed in lattice units and the neighbour taken one
    cell upstream. This is the better choice for strongly unsteady outflow, where coherent
    structures crossing the boundary would otherwise leave a residual imprint; for steady
    or mildly unsteady flow the two give practically the same answer. ``u_conv = 0`` (the
    default) selects plain extrapolation.

    Corners
    -------
    Do not give this condition a node that is also a wall. At a node where an outlet meets
    a no-slip wall, two directions are missing with different natures: the streamwise one
    is open and the wall-normal one is solid. A single per-node condition cannot express
    both. The usual remedy, and the one the XLB tutorials follow, is to build the outlet
    face with ``bounding_box_indices(remove_edges=True)`` so the corner belongs to the
    wall, whose treatment is the more restrictive of the two.

    References
    ----------
    Yang, Z. (2013). Lattice Boltzmann outflow treatments: Convective conditions and
    others. *Computers & Mathematics with Applications*, 65(2), 160-171.

    Ginzburg, I. (2005). Generic boundary conditions for lattice Boltzmann models and
    their application to advection and anisotropic dispersion equations.
    *Advances in Water Resources*, 28(11), 1196-1216.

    Parameters
    ----------
    u_conv : float
        Convection speed for the Sommerfeld variant, in lattice units. Defaults to
        ``0.0``, which selects zeroth-order extrapolation. A sensible non-zero choice is
        the mean velocity through the outlet.
    flow_direction : int
        Axis normal to the outlet, used only by the convective variant to locate the
        upstream neighbour. Defaults to ``0`` (the x axis).
    """

    def __init__(
        self,
        u_conv: float = 0.0,
        flow_direction: int = 0,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
        voxelization_method: MeshVoxelizationMethod = None,
    ):
        self.u_conv = u_conv
        self.flow_direction = flow_direction

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

        # This BC needs padding for finding missing directions when imposed on a geometry
        # that is in the domain interior
        self.needs_padding = True

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        boundary = bc_mask == self.id
        new_shape = (self.velocity_set.q,) + boundary.shape[1:]
        boundary = lax.broadcast_in_dim(boundary, new_shape, tuple(range(self.velocity_set.d + 1)))

        if self.u_conv > 0.0:
            # Convective (Sommerfeld) outflow: relax towards the upstream neighbour.
            # Axis 0 of the population array is the direction index, so the spatial axis
            # normal to the outlet is flow_direction + 1.
            u_conv = self.compute_dtype(self.u_conv)
            f_upstream = jnp.roll(f_post, shift=1, axis=self.flow_direction + 1)
            f_open = (f_pre + u_conv * f_upstream) / (1.0 + u_conv)
        else:
            # Zeroth-order extrapolation: keep the pre-streaming population
            f_open = f_pre

        return jnp.where(jnp.logical_and(missing_mask, boundary), f_open, f_post)

    def _construct_warp(self):
        # Set local constants
        _u_conv = self.compute_dtype(self.u_conv)
        _one = self.compute_dtype(1.0)

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
            # Post-streaming values are only modified at missing directions.
            # Zeroth-order extrapolation keeps the pre-streaming population, which is a
            # purely local operation and therefore the only variant supported on Warp.
            _f = f_post
            for l in range(self.velocity_set.q):
                if missing_mask[l] == wp.uint8(1):
                    _f[l] = f_pre[l]

            return _f

        kernel = self._construct_kernel(functional)

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        if self.u_conv > 0.0:
            raise NotImplementedError("The convective outflow variant is only implemented for the JAX backend; use u_conv = 0.0 on Warp.")
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

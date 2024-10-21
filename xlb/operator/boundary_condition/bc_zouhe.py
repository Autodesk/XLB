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
from xlb.operator.equilibrium import QuadraticEquilibrium


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
        prescribed_value,
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
        self.prescribed_value = prescribed_value

        # Call the parent constructor
        super().__init__(
            ImplementationStep.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
            mesh_vertices,
        )

        # Set the prescribed value for pressure or velocity
        dim = self.velocity_set.d
        if self.compute_backend == ComputeBackend.JAX:
            self.prescribed_value = jnp.atleast_1d(prescribed_value)[(slice(None),) + (None,) * dim]
            # TODO: this won't work if the prescribed values are a profile with the length of bdry indices!

        # This BC needs padding for finding missing directions when imposed on a geometry that is in the domain interior
        self.needs_padding = True

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
            vel = self.prescribed_value
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
            rho = self.prescribed_value
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
        # assign placeholders for both u and rho based on prescribed_value
        _d = self.velocity_set.d
        _q = self.velocity_set.q
        u = self.prescribed_value if self.bc_type == "velocity" else (0,) * _d
        rho = self.prescribed_value if self.bc_type == "pressure" else 0.0

        # Set local constants TODO: This is a hack and should be fixed with warp update
        # _u_vec = wp.vec(_d, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _rho = self.compute_dtype(rho)
        _u = _u_vec(u[0], u[1], u[2]) if _d == 3 else _u_vec(u[0], u[1])
        _opp_indices = self.velocity_set.opp_indices
        _c = self.velocity_set.c
        _c_float = self.velocity_set.c_float
        # TODO: this is way less than ideal. we should not be making new types

        @wp.func
        def _get_fsum(
            fpop: Any,
            missing_mask: Any,
        ):
            fsum_known = self.compute_dtype(0.0)
            fsum_middle = self.compute_dtype(0.0)
            for l in range(_q):
                if missing_mask[_opp_indices[l]] == wp.uint8(1):
                    fsum_known += self.compute_dtype(2.0) * fpop[l]
                elif missing_mask[l] != wp.uint8(1):
                    fsum_middle += fpop[l]
            return fsum_known + fsum_middle

        @wp.func
        def get_normal_vectors(
            missing_mask: Any,
        ):
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) + wp.abs(_c[2, l]) == 1:
                    return -_u_vec(_c_float[0, l], _c_float[1, l], _c_float[2, l])

        @wp.func
        def bounceback_nonequilibrium(
            fpop: Any,
            feq: Any,
            missing_mask: Any,
        ):
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1):
                    fpop[l] = fpop[_opp_indices[l]] + feq[l] - feq[_opp_indices[l]]
            return fpop

        @wp.func
        def functional_velocity(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
        ):
            # Post-streaming values are only modified at missing direction
            _f = f_post

            # Find normal vector
            normals = get_normal_vectors(missing_mask)

            # calculate rho
            fsum = _get_fsum(_f, missing_mask)
            unormal = self.compute_dtype(0.0)
            for d in range(_d):
                unormal += _u[d] * normals[d]
            _rho = fsum / (self.compute_dtype(1.0) + unormal)

            # impose non-equilibrium bounceback
            feq = self.equilibrium_operator.warp_functional(_rho, _u)
            _f = bounceback_nonequilibrium(_f, feq, missing_mask)
            return _f

        @wp.func
        def functional_pressure(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
        ):
            # Post-streaming values are only modified at missing direction
            _f = f_post

            # Find normal vector
            normals = get_normal_vectors(missing_mask)

            # calculate velocity
            fsum = _get_fsum(_f, missing_mask)
            unormal = -self.compute_dtype(1.0) + fsum / _rho
            _u = unormal * normals

            # impose non-equilibrium bounceback
            feq = self.equilibrium_operator.warp_functional(_rho, _u)
            _f = bounceback_nonequilibrium(_f, feq, missing_mask)
            return _f

        if self.bc_type == "velocity":
            functional = functional_velocity
        elif self.bc_type == "pressure":
            functional = functional_pressure
        elif self.bc_type == "velocity":
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

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
from xlb.operator.boundary_condition.bc_zouhe import ZouHeBC
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry
from xlb.operator.macroscopic.second_moment import SecondMoment as MomentumFlux


class RegularizedBC(ZouHeBC):
    """
    Regularized boundary condition for a lattice Boltzmann method simulation.

    This class implements the regularized boundary condition, which is a non-equilibrium bounce-back boundary condition
    with additional regularization. It can be used to set inflow and outflow boundary conditions with prescribed pressure
    or velocity.

    Attributes
    ----------
    name : str
        The name of the boundary condition. For this class, it is "Regularized".
    Qi : numpy.ndarray
        The Qi tensor, which is used in the regularization of the distribution functions.

    References
    ----------
    Latt, J. (2007). Hydrodynamic limit of lattice Boltzmann equations. PhD thesis, University of Geneva.
    Latt, J., Chopard, B., Malaspinas, O., Deville, M., & Michler, A. (2008). Straight velocity boundaries in the
    lattice Boltzmann method. Physical Review E, 77(5), 056703. doi:10.1103/PhysRevE.77.056703
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
        # Call the parent constructor
        super().__init__(
            bc_type,
            prescribed_value,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
            mesh_vertices,
        )
        # Overwrite the boundary condition registry id with the bc_type in the name
        self.momentum_flux = MomentumFlux()

    @partial(jit, static_argnums=(0,), inline=True)
    def regularize_fpop(self, fpop, feq):
        """
        Regularizes the distribution functions by adding non-equilibrium contributions based on second moments of fpop.

        Parameters
        ----------
        fpop : jax.numpy.ndarray
            The distribution functions.
        feq : jax.numpy.ndarray
            The equilibrium distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The regularized distribution functions.
        """
        # Qi = cc - cs^2*I
        dim = self.velocity_set.d
        weights = self.velocity_set.w[(slice(None),) + (None,) * dim]
        Qi = jnp.array(self.velocity_set.qi, dtype=self.compute_dtype)

        # Compute momentum flux of off-equilibrium populations for regularization: Pi^1 = Pi^{neq}
        f_neq = fpop - feq
        PiNeq = self.momentum_flux(f_neq)
        # PiNeq = self.momentum_flux(fpop) - self.momentum_flux(feq)

        # Compute double dot product Qi:Pi1
        # QiPi1 = np.zeros_like(fpop)
        # Pi1 = PiNeq
        QiPi1 = jnp.tensordot(Qi, PiNeq, axes=(1, 0))

        # assign all populations based on eq 45 of Latt et al (2008)
        # fneq ~ f^1
        fpop1 = 9.0 / 2.0 * weights * QiPi1
        fpop_regularized = feq + fpop1
        return fpop_regularized

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

        # Regularize the boundary fpop
        f_post_bd = self.regularize_fpop(f_post_bd, feq)

        # apply bc
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
        # compute Qi tensor and store it in self
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _rho = self.compute_dtype(rho)
        _u = _u_vec(u[0], u[1], u[2]) if _d == 3 else _u_vec(u[0], u[1])
        _opp_indices = self.velocity_set.opp_indices
        _w = self.velocity_set.w
        _c = self.velocity_set.c
        _c_float = self.velocity_set.c_float
        _qi = self.velocity_set.qi
        # TODO: related to _c_float: this is way less than ideal. we should not be making new types

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
        def regularize_fpop(
            fpop: Any,
            feq: Any,
        ):
            """
            Regularizes the distribution functions by adding non-equilibrium contributions based on second moments of fpop.
            """
            # Compute momentum flux of off-equilibrium populations for regularization: Pi^1 = Pi^{neq}
            f_neq = fpop - feq
            PiNeq = self.momentum_flux.warp_functional(f_neq)

            # Compute double dot product Qi:Pi1 (where Pi1 = PiNeq)
            nt = _d * (_d + 1) // 2
            for l in range(_q):
                QiPi1 = self.compute_dtype(0.0)
                for t in range(nt):
                    QiPi1 += _qi[l, t] * PiNeq[t]

                # assign all populations based on eq 45 of Latt et al (2008)
                # fneq ~ f^1
                fpop1 = self.compute_dtype(4.5) * _w[l] * QiPi1
                fpop[l] = feq[l] + fpop1
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

            # Regularize the boundary fpop
            _f = regularize_fpop(_f, feq)
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

            # Regularize the boundary fpop
            _f = regularize_fpop(_f, feq)
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

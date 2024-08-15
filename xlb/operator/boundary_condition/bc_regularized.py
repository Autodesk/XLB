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
from xlb.operator.boundary_condition.bc_zouhe import ZouHeBC
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry
from xlb.operator.macroscopic.second_moment import SecondMoment


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

    id = boundary_condition_registry.register_boundary_condition(__qualname__)

    def __init__(
        self,
        bc_type,
        prescribed_value,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
    ):
        # Call the parent constructor
        super().__init__(
            bc_type,
            prescribed_value,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
        )

        # The operator to compute the momentum flux
        self.momentum_flux = SecondMoment()

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
        Qi = jnp.array(self.velocity_set.cc, dtype=self.compute_dtype)
        if dim == 3:
            diagonal = (0, 3, 5)
            offdiagonal = (1, 2, 4)
        elif dim == 2:
            diagonal = (0, 2)
            offdiagonal = (1,)
        else:
            raise ValueError(f"dim = {dim} not supported")

        # Qi = cc - cs^2*I
        # multiply off-diagonal elements by 2 because the Q tensor is symmetric
        # Qi = Qi.at[:, diagonal].add(-1.0 / 3.0)
        # Qi = Qi.at[:, offdiagonal].multiply(2.0)

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
    def apply_jax(self, f_pre, f_post, boundary_mask, missing_mask):
        # creat a mask to slice boundary cells
        boundary = boundary_mask == self.id
        boundary = jnp.repeat(boundary, self.velocity_set.q, axis=0)

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
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _rho = wp.float32(rho)
        _u = _u_vec(u[0], u[1], u[2]) if _d == 3 else _u_vec(u[0], u[1])
        _opp_indices = self.velocity_set.wp_opp_indices
        _c = self.velocity_set.wp_c
        _c32 = self.velocity_set.wp_c32
        # TODO: this is way less than ideal. we should not be making new types

        @wp.func
        def get_normal_vectors_2d(
            lattice_direction: Any,
        ):
            l = lattice_direction
            if wp.abs(_c[0, l]) + wp.abs(_c[1, l]) == 1:
                normals = -_u_vec(_c32[0, l], _c32[1, l])
            return normals

        @wp.func
        def get_normal_vectors_3d(
            lattice_direction: Any,
        ):
            l = lattice_direction
            if wp.abs(_c[0, l]) + wp.abs(_c[1, l]) + wp.abs(_c[2, l]) == 1:
                normals = -_u_vec(_c32[0, l], _c32[1, l], _c32[2, l])
            return normals

        @wp.func
        def _helper_functional(
            fpop: Any,
            fsum: Any,
            missing_mask: Any,
            lattice_direction: Any,
        ):
            l = lattice_direction
            known_mask = missing_mask[_opp_indices[l]]
            middle_mask = ~(missing_mask[l] | known_mask)
            # fsum += fpop[l] * float(middle_mask) + 2.0 * fpop[l] * float(known_mask)
            if middle_mask and known_mask:
                fsum += fpop[l] + 2.0 * fpop[l]
            elif middle_mask:
                fsum += fpop[l]
            elif known_mask:
                fsum += 2.0 * fpop[l]
            return fsum

        @wp.func
        def bounceback_nonequilibrium(
            fpop: Any,
            missing_mask: Any,
            density: Any,
            velocity: Any,
        ):
            feq = self.equilibrium_operator.warp_functional(density, velocity)
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1):
                    fpop[l] = fpop[_opp_indices[l]] + feq[l] - feq[_opp_indices[l]]
            return fpop

        @wp.func
        def functional3d_velocity(
            f_pre: Any,
            f_post: Any,
            missing_mask: Any,
        ):
            # Post-streaming values are only modified at missing direction
            _f = f_post
            _fsum = self.compute_dtype(0.0)
            unormal = self.compute_dtype(0.0)
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1):
                    normals = get_normal_vectors_3d(l)
                _fsum = _helper_functional(_f, _fsum, missing_mask, l)

            for d in range(_d):
                unormal += _u[d] * normals[d]
            _rho = _fsum / (1.0 + unormal)

            # impose non-equilibrium bounceback
            _f = bounceback_nonequilibrium(_f, missing_mask, _rho, _u)
            return _f

        @wp.func
        def functional3d_pressure(
            f_pre: Any,
            f_post: Any,
            missing_mask: Any,
        ):
            # Post-streaming values are only modified at missing direction
            _f = f_post
            _fsum = self.compute_dtype(0.0)
            unormal = self.compute_dtype(0.0)
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1):
                    normals = get_normal_vectors_3d(l)
                _fsum = _helper_functional(_f, _fsum, missing_mask, l)

            unormal = -1.0 + _fsum / _rho
            _u = unormal * normals

            # impose non-equilibrium bounceback
            _f = bounceback_nonequilibrium(_f, missing_mask, _rho, _u)
            return _f

        @wp.func
        def functional2d_velocity(
            f_pre: Any,
            f_post: Any,
            missing_mask: Any,
        ):
            # Post-streaming values are only modified at missing direction
            _f = f_post
            _fsum = self.compute_dtype(0.0)
            unormal = self.compute_dtype(0.0)
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1):
                    normals = get_normal_vectors_2d(l)
                _fsum = _helper_functional(_f, _fsum, missing_mask, l)

            for d in range(_d):
                unormal += _u[d] * normals[d]
            _rho = _fsum / (1.0 + unormal)

            # impose non-equilibrium bounceback
            _f = bounceback_nonequilibrium(_f, missing_mask, _rho, _u)
            return _f

        @wp.func
        def functional2d_pressure(
            f_pre: Any,
            f_post: Any,
            missing_mask: Any,
        ):
            # Post-streaming values are only modified at missing direction
            _f = f_post
            _fsum = self.compute_dtype(0.0)
            unormal = self.compute_dtype(0.0)
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1):
                    normals = get_normal_vectors_2d(l)
                _fsum = _helper_functional(_f, _fsum, missing_mask, l)

            unormal = -1.0 + _fsum / _rho
            _u = unormal * normals

            # impose non-equilibrium bounceback
            _f = bounceback_nonequilibrium(_f, missing_mask, _rho, _u)
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
        if self.velocity_set.d == 3 and self.bc_type == "velocity":
            functional = functional3d_velocity
        elif self.velocity_set.d == 3 and self.bc_type == "pressure":
            functional = functional3d_pressure
        elif self.bc_type == "velocity":
            functional = functional2d_velocity
        else:
            functional = functional2d_pressure

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

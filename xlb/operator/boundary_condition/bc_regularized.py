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
from xlb.operator.boundary_condition.bc_zouhe import ZouHeBC
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
        profile=None,
        prescribed_value: Union[float, Tuple[float, ...], np.ndarray] = None,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
    ):
        # Call the parent constructor
        super().__init__(
            bc_type,
            profile,
            prescribed_value,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
            mesh_vertices,
        )
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
        # load helper functions
        from xlb.helper.bc_warp_functions import get_normal_vectors, get_bc_fsum, bounceback_nonequilibrium, regularize_fpop

        # Set local constants
        _d = self.velocity_set.d
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices

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

            # Find the value of u from the missing directions
            for l in range(_q):
                # Since we are only considering normal velocity, we only need to find one value
                if missing_mask[l] == wp.uint8(1):
                    # Create velocity vector by multiplying the prescribed value with the normal vector
                    prescribed_value = f_1[_opp_indices[l], index[0], index[1], index[2]]
                    _u = -prescribed_value * normals
                    break

            # calculate rho
            fsum = get_bc_fsum(_f, missing_mask)
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

            # Find the value of rho from the missing directions
            for q in range(_q):
                # Since we need only one scalar value, we only need to find one value
                if missing_mask[q] == wp.uint8(1):
                    _rho = f_1[_opp_indices[q], index[0], index[1], index[2]]
                    break

            # calculate velocity
            fsum = get_bc_fsum(_f, missing_mask)
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

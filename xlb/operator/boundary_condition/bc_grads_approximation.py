"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit
import jax.lax as lax
from functools import partial
import warp as wp
from typing import Any
from collections import Counter
import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.macroscopic.zero_moment import ZeroMoment
from xlb.operator.macroscopic.second_moment import SecondMoment as MomentumFlux
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.boundary_condition.boundary_condition import (
    ImplementationStep,
    BoundaryCondition,
)
from xlb.operator.boundary_condition.boundary_condition_registry import (
    boundary_condition_registry,
)


class GradsApproximationBC(BoundaryCondition):
    """
    Purpose: Using Grad's approximation to represent fpop based on macroscopic inputs used for outflow [1] and
    Dirichlet BCs [2]
    [1] S. Chikatamarla, S. Ansumali, and I. Karlin, "Grad's approximation for missing data in lattice Boltzmann
        simulations", Europhys. Lett. 74, 215 (2006).
    [2] Dorschner, B., Chikatamarla, S. S., Bösch, F., & Karlin, I. V. (2015). Grad's approximation for moving and
        stationary walls in entropic lattice Boltzmann simulations. Journal of Computational Physics, 295, 340-354.

    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
    ):
        # TODO: the input velocity must be suitably stored elesewhere when mesh is moving.
        self.u = (0, 0, 0)

        # Call the parent constructor
        super().__init__(
            ImplementationStep.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
            mesh_vertices,
        )

        # Instantiate the operator for computing macroscopic values
        self.macroscopic = Macroscopic()
        self.zero_moment = ZeroMoment()
        self.equilibrium = QuadraticEquilibrium()
        self.momentum_flux = MomentumFlux()

        # This BC needs implicit distance to the mesh
        self.needs_mesh_distance = True

        # If this BC is defined using indices, it would need padding in order to find missing directions
        # when imposed on a geometry that is in the domain interior
        if self.mesh_vertices is None:
            assert self.indices is not None
            self.needs_padding = True

        # Raise error if used for 2d examples:
        if self.velocity_set.d == 2:
            raise NotImplementedError("This BC is not implemented in 2D!")

        # if indices is not None:
        #     # this BC would be limited to stationary boundaries
        #     # assert mesh_vertices is None
        # if mesh_vertices is not None:
        #     # this BC would be applicable for stationary and moving boundaries
        #     assert indices is None
        #     if mesh_velocity_function is not None:
        #         # mesh is moving and/or deforming

        assert self.compute_backend == ComputeBackend.WARP, "This BC is currently only implemented with the Warp backend!"

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        # TODO
        raise NotImplementedError(f"Operation {self.__class__.__name} not implemented in JAX!")
        return

    def _construct_warp(self):
        # Set local variables and constants
        _c = self.velocity_set.c
        _q = self.velocity_set.q
        _d = self.velocity_set.d
        _w = self.velocity_set.w
        _qi = self.velocity_set.qi
        _opp_indices = self.velocity_set.opp_indices
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _u_wall = _u_vec(self.u[0], self.u[1], self.u[2]) if _d == 3 else _u_vec(self.u[0], self.u[1])
        # diagonal = wp.vec3i(0, 3, 5) if _d == 3 else wp.vec2i(0, 2)

        @wp.func
        def regularize_fpop(
            missing_mask: Any,
            rho: Any,
            u: Any,
            fpop: Any,
        ):
            """
            Regularizes the distribution functions by adding non-equilibrium contributions based on second moments of fpop.
            """
            # Compute momentum flux of off-equilibrium populations for regularization: Pi^1 = Pi^{neq}
            feq = self.equilibrium.warp_functional(rho, u)
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
        def grads_approximate_fpop(
            missing_mask: Any,
            rho: Any,
            u: Any,
            f_post: Any,
        ):
            # Purpose: Using Grad's approximation to represent fpop based on macroscopic inputs used for outflow [1] and
            # Dirichlet BCs [2]
            # [1] S. Chikatax`marla, S. Ansumali, and I. Karlin, "Grad's approximation for missing data in lattice Boltzmann
            #   simulations", Europhys. Lett. 74, 215 (2006).
            # [2] Dorschner, B., Chikatamarla, S. S., Bösch, F., & Karlin, I. V. (2015). Grad's approximation for moving and
            #    stationary walls in entropic lattice Boltzmann simulations. Journal of Computational Physics, 295, 340-354.

            # Note: See also self.regularize_fpop function which is somewhat similar.

            # Compute pressure tensor Pi using all f_post-streaming values
            Pi = self.momentum_flux.warp_functional(f_post)

            # Compute double dot product Qi:Pi1 (where Pi1 = PiNeq)
            nt = _d * (_d + 1) // 2
            for l in range(_q):
                # if missing_mask[l] == wp.uint8(1):
                QiPi = self.compute_dtype(0.0)
                for t in range(nt):
                    if t == 0 or t == 3 or t == 5:
                        QiPi += _qi[l, t] * (Pi[t] - rho / self.compute_dtype(3.0))
                    else:
                        QiPi += _qi[l, t] * Pi[t]

                # Compute c.u
                cu = self.compute_dtype(0.0)
                for d in range(self.velocity_set.d):
                    if _c[d, l] == 1:
                        cu += u[d]
                    elif _c[d, l] == -1:
                        cu -= u[d]
                cu *= self.compute_dtype(3.0)

                # change f_post using the Grad's approximation
                f_post[l] = rho * _w[l] * (self.compute_dtype(1.0) + cu) + _w[l] * self.compute_dtype(4.5) * QiPi

            return f_post

        # Construct the functionals for this BC
        @wp.func
        def functional_method1(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
        ):
            # NOTE: this BC has been reformulated to become entirely local and so has differences compared to the original paper.
            #       Here we use the current time-step populations (f_pre = f_post_collision and f_post = f_post_streaming).
            one = self.compute_dtype(1.0)
            for l in range(_q):
                # If the mask is missing then take the opposite index
                if missing_mask[l] == wp.uint8(1):
                    # The implicit distance to the boundary or "weights" have been stored in known directions of f_1
                    # weight = f_1[_opp_indices[l], index[0], index[1], index[2]]
                    weight = self.compute_dtype(0.5)

                    # Use differentiable interpolated BB to find f_missing:
                    f_post[l] = ((one - weight) * f_post[_opp_indices[l]] + weight * (f_pre[l] + f_pre[_opp_indices[l]])) / (one + weight)

                    # # Add contribution due to moving_wall to f_missing as is usual in regular Bouzidi BC
                    # cu = self.compute_dtype(0.0)
                    # for d in range(_d):
                    #     if _c[d, l] == 1:
                    #         cu += _u_wall[d]
                    #     elif _c[d, l] == -1:
                    #         cu -= _u_wall[d]
                    # cu *= self.compute_dtype(-6.0) * _w[l]
                    # f_post[l] += cu

            # Compute density, velocity using all f_post-streaming values
            rho, u = self.macroscopic.warp_functional(f_post)

            # Compute Grad's appriximation using full equation as in Eq (10) of Dorschner et al.
            f_post = regularize_fpop(missing_mask, rho, u, f_post)
            # f_post = grads_approximate_fpop(missing_mask, rho, u, f_post)
            return f_post

        # Construct the functionals for this BC
        @wp.func
        def functional_method2(
            index: Any,
            timestep: Any,
            missing_mask: Any,
            f_0: Any,
            f_1: Any,
            f_pre: Any,
            f_post: Any,
        ):
            # NOTE: this BC has been reformulated to become entirely local and so has differences compared to the original paper.
            #       Here we use the current time-step populations (f_pre = f_post_collision and f_post = f_post_streaming).
            # NOTE: f_aux should contain populations at "x_f" (see their fig 1) in the missign direction of the BC which amounts
            #       to post-collision values being pulled from appropriate cells like ExtrapolationBC
            #
            # here I need to compute all terms in Eq (10)
            # Strategy:
            # 1) "weights" should have been stored somewhere to be used here.
            # 2) Given "weights", "u_w" (input to the BC) and "u_f" (computed from f_aux), compute "u_target" as per Eq (14)
            #    NOTE: in the original paper "u_target" is associated with the previous time step not current time.
            # 3) Given "weights" use differentiable interpolated BB to find f_missing as I had before:
            # fmissing = ((1. - weights) * f_poststreaming_iknown + weights * (f_postcollision_imissing + f_postcollision_iknown)) / (1.0 + weights)
            # 4) Add contribution due to u_w to f_missing as is usual in regular Bouzidi BC (ie. -6.0 * self.lattice.w * jnp.dot(self.vel, c)
            # 5) Compute rho_target = \sum(f_ibb) based on these values
            # 6) Compute feq using feq = self.equilibrium(rho_target, u_target)
            # 7) Compute Pi_neq and Pi_eq using all f_post-streaming values as per:
            #       Pi_neq = self.momentum_flux(fneq) and Pi_eq = self.momentum_flux(feq)
            # 8) Compute Grad's appriximation using full equation as in Eq (10)
            #    NOTE: this is very similar to the regularization procedure.

            _f_nbr = _f_vec()
            u_target = _u_vec(0.0, 0.0, 0.0) if _d == 3 else _u_vec(0.0, 0.0)
            num_missing = 0
            one = self.compute_dtype(1.0)
            for l in range(_q):
                # If the mask is missing then take the opposite index
                if missing_mask[l] == wp.uint8(1):
                    # Find the neighbour and its velocity value
                    for ll in range(_q):
                        # f_0 is the post-collision values of the current time-step
                        # Get index associated with the fluid neighbours
                        fluid_nbr_index = type(index)()
                        for d in range(_d):
                            fluid_nbr_index[d] = index[d] + _c[d, l]
                        # The following is the post-collision values of the fluid neighbor cell
                        _f_nbr[ll] = self.compute_dtype(f_0[ll, fluid_nbr_index[0], fluid_nbr_index[1], fluid_nbr_index[2]])

                    # Compute the velocity vector at the fluid neighbouring cells
                    _, u_f = self.macroscopic.warp_functional(_f_nbr)

                    # Record the number of missing directions
                    num_missing += 1

                    # The implicit distance to the boundary or "weights" have been stored in known directions of f_1
                    weight = f_1[_opp_indices[l], index[0], index[1], index[2]]

                    # Given "weights", "u_w" (input to the BC) and "u_f" (computed from f_aux), compute "u_target" as per Eq (14)
                    for d in range(_d):
                        u_target[d] += (weight * u_f[d] + _u_wall[d]) / (one + weight)

                    # Use differentiable interpolated BB to find f_missing:
                    f_post[l] = ((one - weight) * f_post[_opp_indices[l]] + weight * (f_pre[l] + f_pre[_opp_indices[l]])) / (one + weight)

                    # Add contribution due to moving_wall to f_missing as is usual in regular Bouzidi BC
                    cu = self.compute_dtype(0.0)
                    for d in range(_d):
                        if _c[d, l] == 1:
                            cu += _u_wall[d]
                        elif _c[d, l] == -1:
                            cu -= _u_wall[d]
                    cu *= self.compute_dtype(-6.0) * _w[l]
                    f_post[l] += cu

            # Compute rho_target = \sum(f_ibb) based on these values
            rho_target = self.zero_moment.warp_functional(f_post)
            for d in range(_d):
                u_target[d] /= num_missing

            # Compute Grad's appriximation using full equation as in Eq (10) of Dorschner et al.
            f_post = grads_approximate_fpop(missing_mask, rho_target, u_target, f_post)
            return f_post

        functional = functional_method1

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
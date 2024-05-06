# Base class for all stepper operators

from logging import warning
from functools import partial
from jax import jit
import warp as wp
from typing import Any

from xlb.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.stepper import Stepper
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep


class IncompressibleNavierStokesStepper(Stepper):
    """
    Class that handles the construction of lattice boltzmann stepping operator
    """

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0, 4), donate_argnums=(1))
    def jax_implementation(self, f_0, f_1, boundary_id_field, missing_mask, timestep):
        """
        Perform a single step of the lattice boltzmann method
        """

        # Cast to compute precision
        f_0 = self.precision_policy.cast_to_compute_jax(f_0)
        f_1 = self.precision_policy.cast_to_compute_jax(f_1)

        # Compute the macroscopic variables
        rho, u = self.macroscopic(f_0)

        # Compute equilibrium
        feq = self.equilibrium(rho, u)

        # Apply collision
        f_post_collision = self.collision(
            f_0,
            feq,
            rho,
            u,
        )

        # Apply collision type boundary conditions
        for bc in self.boundary_conditions:
            if bc.implementation_step == ImplementationStep.COLLISION:
                f_0 = bc(
                    f_0,
                    f_post_collision,
                    boundary_id_field,
                    missing_mask,
                )

        # Apply streaming
        f_1 = self.stream(f_0)

        # Apply boundary conditions
        for bc in self.boundary_conditions:
            if bc.implementation_step == ImplementationStep.STREAMING:
                f_1 = bc(
                    f_post_collision,
                    f_1,
                    boundary_id_field,
                    missing_mask,
                )

        # Copy back to store precision
        f_1 = self.precision_policy.cast_to_store_jax(f_1)

        return f_1

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(
            self.velocity_set.q, dtype=wp.uint8
        )  # TODO fix vec bool

        # Get the boundary condition ids
        _equilibrium_bc = wp.uint8(self.equilibrium_bc.id)
        _do_nothing_bc = wp.uint8(self.do_nothing_bc.id)
        _halfway_bounce_back_bc = wp.uint8(self.halfway_bounce_back_bc.id)
        _fullway_bounce_back_bc = wp.uint8(self.fullway_bounce_back_bc.id)

        # Construct the kernel
        @wp.kernel
        def kernel(
            f_0: wp.array4d(dtype=Any),
            f_1: wp.array4d(dtype=Any),
            boundary_id_field: wp.array4d(dtype=Any),
            missing_mask: wp.array4d(dtype=Any),
            timestep: int,
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)  # TODO warp should fix this

            # Get the boundary id and missing mask
            _boundary_id = boundary_id_field[0, index[0], index[1], index[2]]
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # TODO fix vec bool
                if missing_mask[l, index[0], index[1], index[2]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)

            # Apply streaming boundary conditions
            if (_boundary_id == wp.uint8(0)) or _boundary_id == _fullway_bounce_back_bc:
                # Regular streaming
                f_post_stream = self.stream.warp_functional(f_0, index)
            elif _boundary_id == _equilibrium_bc:
                # Equilibrium boundary condition
                f_post_stream = self.equilibrium_bc.warp_functional(
                    f_0, _missing_mask, index
                )
            elif _boundary_id == _do_nothing_bc:
                # Do nothing boundary condition
                f_post_stream = self.do_nothing_bc.warp_functional(
                    f_0, _missing_mask, index
                )
            elif _boundary_id == _halfway_bounce_back_bc:
                # Half way boundary condition
                f_post_stream = self.halfway_bounce_back_bc.warp_functional(
                    f_0, _missing_mask, index
                )

            # Compute rho and u
            rho, u = self.macroscopic.warp_functional(f_post_stream)

            # Compute equilibrium
            feq = self.equilibrium.warp_functional(rho, u)

            # Apply collision
            f_post_collision = self.collision.warp_functional(
                f_post_stream,
                feq,
                rho,
                u,
            )

            # Apply collision type boundary conditions
            if _boundary_id == _fullway_bounce_back_bc:
                # Full way boundary condition
                f_post_collision = self.fullway_bounce_back_bc.warp_functional(
                    f_post_stream,
                    f_post_collision,
                    _missing_mask,
                )

            # Set the output
            for l in range(self.velocity_set.q):
                f_1[l, index[0], index[1], index[2]] = f_post_collision[l]

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_0, f_1, boundary_id_field, missing_mask, timestep):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f_0,
                f_1,
                boundary_id_field,
                missing_mask,
                timestep,
            ],
            dim=f_0.shape[1:],
        )
        return f_1

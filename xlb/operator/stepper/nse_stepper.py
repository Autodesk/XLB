# Base class for all stepper operators

from functools import partial
from jax import jit
import warp as wp
from typing import Any

from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.stream import Stream
from xlb.operator.collision import BGK, KBC
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.stepper import Stepper
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep


class IncompressibleNavierStokesStepper(Stepper):
    def __init__(self, omega, boundary_conditions=[], collision_type="BGK"):
        velocity_set = DefaultConfig.velocity_set
        precision_policy = DefaultConfig.default_precision_policy
        compute_backend = DefaultConfig.default_backend

        # Construct the collision operator
        if collision_type == "BGK":
            self.collision = BGK(omega, velocity_set, precision_policy, compute_backend)
        elif collision_type == "KBC":
            self.collision = KBC(omega, velocity_set, precision_policy, compute_backend)

        # Construct the operators
        self.stream = Stream(velocity_set, precision_policy, compute_backend)
        self.equilibrium = QuadraticEquilibrium(velocity_set, precision_policy, compute_backend)
        self.macroscopic = Macroscopic(velocity_set, precision_policy, compute_backend)

        operators = [self.macroscopic, self.equilibrium, self.collision, self.stream]

        super().__init__(operators, boundary_conditions)

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_0, f_1, boundary_mask, missing_mask, timestep):
        """
        Perform a single step of the lattice boltzmann method
        """
        # Cast to compute precisioni
        f_0 = self.precision_policy.cast_to_compute_jax(f_0)
        f_1 = self.precision_policy.cast_to_compute_jax(f_1)

        # Compute the macroscopic variables
        rho, u = self.macroscopic(f_0)

        # Compute equilibrium
        feq = self.equilibrium(rho, u)

        # Apply collision
        f_post_collision = self.collision(f_0, feq, rho, u)

        # Apply collision type boundary conditions
        for bc in self.boundary_conditions:
            if bc.implementation_step == ImplementationStep.COLLISION:
                f_post_collision = bc(
                    f_0,
                    f_post_collision,
                    boundary_mask,
                    missing_mask,
                )

        # Apply streaming
        f_1 = self.stream(f_post_collision)

        # Apply boundary conditions
        for bc in self.boundary_conditions:
            if bc.implementation_step == ImplementationStep.STREAMING:
                f_1 = bc(
                    f_post_collision,
                    f_1,
                    boundary_mask,
                    missing_mask,
                )

        # Copy back to store precision
        f_1 = self.precision_policy.cast_to_store_jax(f_1)

        return f_1

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)  # TODO fix vec bool

        # Get the boundary condition ids
        _equilibrium_bc = wp.uint8(self.equilibrium_bc.id)
        _do_nothing_bc = wp.uint8(self.do_nothing_bc.id)
        _halfway_bounce_back_bc = wp.uint8(self.halfway_bounce_back_bc.id)
        _fullway_bounce_back_bc = wp.uint8(self.fullway_bounce_back_bc.id)

        @wp.kernel
        def kernel2d(
            f_0: wp.array3d(dtype=Any),
            f_1: wp.array3d(dtype=Any),
            boundary_mask: wp.array3d(dtype=Any),
            missing_mask: wp.array3d(dtype=Any),
            timestep: int,
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)  # TODO warp should fix this

            # Get the boundary id and missing mask
            _boundary_id = boundary_mask[0, index[0], index[1]]
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # TODO fix vec bool
                if missing_mask[l, index[0], index[1]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)

            # Apply streaming boundary conditions
            if (_boundary_id == wp.uint8(0)) or _boundary_id == _fullway_bounce_back_bc:
                # Regular streaming
                f_post_stream = self.stream.warp_functional(f_0, index)
            elif _boundary_id == _equilibrium_bc:
                # Equilibrium boundary condition
                f_post_stream = self.equilibrium_bc.warp_functional(f_0, _missing_mask, index)
            elif _boundary_id == _do_nothing_bc:
                # Do nothing boundary condition
                f_post_stream = self.do_nothing_bc.warp_functional(f_0, _missing_mask, index)
            elif _boundary_id == _halfway_bounce_back_bc:
                # Half way boundary condition
                f_post_stream = self.halfway_bounce_back_bc.warp_functional(f_0, _missing_mask, index)

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
                f_1[l, index[0], index[1]] = f_post_collision[l]

        # Construct the kernel
        @wp.kernel
        def kernel3d(
            f_0: wp.array4d(dtype=Any),
            f_1: wp.array4d(dtype=Any),
            boundary_mask: wp.array4d(dtype=Any),
            missing_mask: wp.array4d(dtype=Any),
            timestep: int,
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)  # TODO warp should fix this

            # Get the boundary id and missing mask
            _boundary_id = boundary_mask[0, index[0], index[1], index[2]]
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
                f_post_stream = self.equilibrium_bc.warp_functional(f_0, _missing_mask, index)
            elif _boundary_id == _do_nothing_bc:
                # Do nothing boundary condition
                f_post_stream = self.do_nothing_bc.warp_functional(f_0, _missing_mask, index)
            elif _boundary_id == _halfway_bounce_back_bc:
                # Half way boundary condition
                f_post_stream = self.halfway_bounce_back_bc.warp_functional(f_0, _missing_mask, index)

            # Compute rho and u
            rho, u = self.macroscopic.warp_functional(f_post_stream)

            # Compute equilibrium
            feq = self.equilibrium.warp_functional(rho, u)

            # Apply collision
            f_post_collision = self.collision.warp_functional(f_post_stream, feq, rho, u)

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

        # Return the correct kernel
        kernel = kernel3d if self.velocity_set.d == 3 else kernel2d

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_0, f_1, boundary_mask, missing_mask, timestep):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f_0,
                f_1,
                boundary_mask,
                missing_mask,
                timestep,
            ],
            dim=f_0.shape[1:],
        )
        return f_1

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
    def apply_jax(self, f, boundary_id, missing_mask, timestep):
        """
        Perform a single step of the lattice boltzmann method
        """

        # Cast to compute precision TODO add this back in
        #f_pre_collision = self.precision_policy.cast_to_compute_jax(f)

        # Compute the macroscopic variables
        rho, u = self.macroscopic(f)

        # Compute equilibrium
        feq = self.equilibrium(rho, u)

        # Apply collision
        f_post_collision = self.collision(
            f,
            feq,
            rho,
            u,
        )

        # Apply collision type boundary conditions
        for bc in self.boundary_conditions:
            if bc.implementation_step == ImplementationStep.COLLISION:
                f_post_collision = bc(
                    f,
                    f_post_collision,
                    boundary_id,
                    missing_mask,
                )

        ## Apply forcing
        # if self.forcing_op is not None:
        #    f = self.forcing_op.apply_jax(f, timestep)

        # Apply streaming
        f_post_streaming = self.stream(f_post_collision)

        # Apply boundary conditions
        for bc in self.boundary_conditions:
            if bc.implementation_step == ImplementationStep.STREAMING:
                f_post_streaming = bc(
                    f_post_collision,
                    f_post_streaming,
                    boundary_id,
                    missing_mask,
                )

        # Copy back to store precision
        #f = self.precision_policy.cast_to_store_jax(f_post_streaming)

        return f_post_streaming

    @Operator.register_backend(ComputeBackend.PALLAS)
    @partial(jit, static_argnums=(0,))
    def apply_pallas(self, fin, boundary_id, missing_mask, timestep):
        # Raise warning that the boundary conditions are not implemented
        warning("Boundary conditions are not implemented for PALLAS backend currently")

        from xlb.operator.parallel_operator import ParallelOperator

        def _pallas_collide(fin, fout):
            idx = pl.program_id(0)

            f = pl.load(fin, (slice(None), idx, slice(None), slice(None)))

            print("f shape", f.shape)

            rho, u = self.macroscopic(f)

            print("rho shape", rho.shape)
            print("u shape", u.shape)

            feq = self.equilibrium(rho, u)

            print("feq shape", feq.shape)

            for i in range(self.velocity_set.q):
                print("f shape", f[i].shape)
                f_post_collision = self.collision(f[i], feq[i])
                print("f_post_collision shape", f_post_collision.shape)
                pl.store(fout, (i, idx, slice(None), slice(None)), f_post_collision)
            # f_post_collision = self.collision(f, feq)
            # pl.store(fout, (i, idx, slice(None), slice(None)), f_post_collision)

        @jit
        def _pallas_collide_kernel(fin):
            return pl.pallas_call(
                partial(_pallas_collide),
                out_shape=jax.ShapeDtypeStruct(
                    ((self.velocity_set.q,) + (self.grid.grid_shape_per_gpu)), fin.dtype
                ),
                # grid=1,
                grid=(self.grid.grid_shape_per_gpu[0], 1, 1),
            )(fin)

        def _pallas_collide_and_stream(f):
            f = _pallas_collide_kernel(f)
            # f = self.stream._streaming_jax_p(f)

            return f

        fout = ParallelOperator(
            self.grid, _pallas_collide_and_stream, self.velocity_set
        )(fin)

        return fout

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
            boundary_id: wp.array4d(dtype=Any),
            missing_mask: wp.array4d(dtype=Any),
            timestep: int,
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)  # TODO warp should fix this

            # Get the boundary id and missing mask
            _boundary_id = boundary_id[0, index[0], index[1], index[2]]
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # TODO fix vec bool
                if missing_mask[l, index[0], index[1], index[2]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)

            # Apply streaming boundary conditions
            if _boundary_id == _equilibrium_bc:
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
            else:
                # Regular streaming
                f_post_stream = self.stream.warp_functional(f_0, index)
 
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
    def warp_implementation(self, f_0, f_1, boundary_id, missing_mask, timestep):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f_0,
                f_1,
                boundary_id,
                missing_mask,
                timestep,
            ],
            dim=f_0.shape[1:],
        )
        return f_1

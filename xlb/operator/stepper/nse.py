# Base class for all stepper operators

from logging import warning
from functools import partial
from jax import jit
import warp as wp

from xlb.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.stepper import Stepper
from xlb.operator.boundary_condition import ImplementationStep
from xlb.operator.collision import BGK


class IncompressibleNavierStokesStepper(Stepper):
    """
    Class that handles the construction of lattice boltzmann stepping operator
    """

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0, 5))
    def apply_jax(self, f, boundary_id, mask, timestep):
        """
        Perform a single step of the lattice boltzmann method
        """

        # Cast to compute precision
        f_pre_collision = self.precision_policy.cast_to_compute_jax(f)

        # Compute the macroscopic variables
        rho, u = self.macroscopic(f_pre_collision)

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
        for id_number, bc in self.collision_boundary_conditions.items():
            f_post_collision = bc(
                f_pre_collision,
                f_post_collision,
                boundary_id == id_number,
                mask,
            )
        f_pre_streaming = f_post_collision

        ## Apply forcing
        # if self.forcing_op is not None:
        #    f = self.forcing_op.apply_jax(f, timestep)

        # Apply streaming
        f_post_streaming = self.stream(f_pre_streaming)

        # Apply boundary conditions
        for id_number, bc in self.stream_boundary_conditions.items():
            f_post_streaming = bc(
                f_pre_streaming,
                f_post_streaming,
                boundary_id == id_number,
                mask,
            )

        # Copy back to store precision
        f = self.precision_policy.cast_to_store_jax(f_post_streaming)

        return f

    @Operator.register_backend(ComputeBackend.PALLAS)
    @partial(jit, static_argnums=(0,))
    def apply_pallas(self, fin, boundary_id, mask, timestep):
        # Raise warning that the boundary conditions are not implemented
        ################################################################
        warning("Boundary conditions are not implemented for PALLAS backend currently")
        ################################################################

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
        # Make constants for warp
        _c = wp.constant(self._warp_stream_mat(self.velocity_set.c))
        _q = wp.constant(self.velocity_set.q)
        _d = wp.constant(self.velocity_set.d)
        _nr_boundary_conditions = wp.constant(len(self.boundary_conditions))

        # Construct the kernel
        @wp.kernel
        def kernel(
            f_0: self._warp_array_type,
            f_1: self._warp_array_type,
            boundary_id: self._warp_uint8_array_type,
            mask: self._warp_bool_array_type,
            timestep: int,
            max_i: int,
            max_j: int,
            max_k: int,
        ):
            # Get the global index
            i, j, k = wp.tid()

            # Get the f, boundary id and mask
            _f = self._warp_lattice_vec()
            _boundary_id = boundary_id[0, i, j, k]
            _mask = self._warp_bool_lattice_vec()
            for l in range(_q):
                _f[l] = f_0[l, i, j, k]

                # TODO fix vec bool
                if mask[l, i, j, k]:
                    _mask[l] = wp.uint8(1)
                else:
                    _mask[l] = wp.uint8(0)

            # Compute rho and u
            rho, u = self.macroscopic.warp_functional(_f)

            # Compute equilibrium
            feq = self.equilibrium.warp_functional(rho, u)

            # Apply collision
            f_post_collision = self.collision.warp_functional(
                _f,
                feq,
                rho,
                u,
            )

            ## Apply collision type boundary conditions
            #if _boundary_id != wp.uint8(0):
            #    f_post_collision = self.collision_boundary_conditions[
            #        _boundary_id
            #    ].warp_functional(
            #        _f,
            #        f_post_collision,
            #        _mask,
            #    )
            f_pre_streaming = f_post_collision  # store pre streaming vector

            # Apply forcing
            # if self.forcing_op is not None:
            #    f = self.forcing.warp_functional(f, timestep)

            # Apply streaming
            for l in range(_q):
                # Get the streamed indices
                streamed_i, streamed_j, streamed_k = self.stream.warp_functional(
                    l, i, j, k, max_i, max_j, max_k
                )
                streamed_l = l

                ## Modify the streamed indices based on streaming boundary condition
                # if _boundary_id != 0:
                #    streamed_l, streamed_i, streamed_j, streamed_k = self.stream_boundary_conditions[id_number].warp_functional(
                #        streamed_l, streamed_i, streamed_j, streamed_k, self._warp_max_i, self._warp_max_j, self._warp_max_k
                #    )

                # Set the output
                f_1[streamed_l, streamed_i, streamed_j, streamed_k] = f_pre_streaming[l]

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_0, f_1, boundary_id, mask, timestep):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f_0,
                f_1,
                boundary_id,
                mask,
                timestep,
                f_0.shape[1],
                f_0.shape[2],
                f_0.shape[3],
            ],
            dim=f_0.shape[1:],
        )
        return f_1

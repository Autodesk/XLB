from functools import partial
import jax.numpy as jnp
from jax import jit, lax
import warp as wp
from typing import Any
from enum import Enum, auto

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.stream import Stream


# Enum used to keep track of LBM operations
class LBMOperationSequence(Enum):
    """
    Note that for dense and single resolution simulations in XLB, the order of operations in the stepper is "stream-then-collide".
    For MultiRes stepper however the order of operations is always "collide-then-stream" except at the finest level when the FUSION_AT_FINEST
    optimization is used.
    In that case the order of operations is "stream-then-collide" ONLY at the finest level.
    """

    STREAM_THEN_COLLIDE = auto()
    COLLIDE_THEN_STREAM = auto()


class FetchPopulations(Operator):
    """
    This operator is used to get the post-collision and post-streaming populations
    Note that for dense and single resolution simulations in XLB, the order of operations in the stepper is "stream-then-collide".
    Therefore, f_0 represents the post-collision values and post_streaming values of the current time step need to be reconstructed
    by applying the streaming and boundary conditions. These populations are readily available in XLB when using multi-resolution
    grids because the mres stepper relies on "collide-then-stream".
    """

    def __init__(
        self,
        no_slip_bc_instance,
        operation_sequence: LBMOperationSequence = LBMOperationSequence.STREAM_THEN_COLLIDE,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        self.no_slip_bc_instance = no_slip_bc_instance
        self.stream = Stream(velocity_set, precision_policy, compute_backend)
        self.operation_sequence = operation_sequence

        if compute_backend == ComputeBackend.WARP:
            self.stream_functional = self.stream.warp_functional
            self.bc_functional = self.no_slip_bc_instance.warp_functional
        elif compute_backend == ComputeBackend.NEON:
            self.stream_functional = self.stream.neon_functional
            self.bc_functional = self.no_slip_bc_instance.neon_functional

        # Call the parent constructor
        super().__init__(
            velocity_set,
            precision_policy,
            compute_backend,
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_0, f_1, bc_mask, missing_mask):
        # Give the input post-collision populations, streaming once and apply the BC the find post-stream values.
        f_post_collision = f_0
        f_post_stream = self.stream(f_post_collision)
        f_post_stream = self.no_slip_bc_instance(f_post_collision, f_post_stream, bc_mask, missing_mask)
        return f_post_collision, f_post_stream

    def _construct_warp(self):
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)

        @wp.func
        def functional_stream_then_collide(
            index: Any,
            f_0: Any,
            f_1: Any,
            _missing_mask: Any,
        ):
            # Get the distribution function
            f_post_collision = _f_vec()
            for l in range(self.velocity_set.q):
                f_post_collision[l] = self.compute_dtype(self.read_field(f_0, index, l))

            # Apply streaming (pull method)
            timestep = 0
            f_post_stream = self.stream_functional(f_0, index)
            f_post_stream = self.bc_functional(index, timestep, _missing_mask, f_0, f_1, f_post_collision, f_post_stream)
            return f_post_collision, f_post_stream

        @wp.func
        def functional_collide_then_stream(
            index: Any,
            f_0: Any,
            f_1: Any,
            _missing_mask: Any,
        ):
            # Get the distribution function
            f_post_collision = _f_vec()
            f_post_stream = _f_vec()
            for l in range(self.velocity_set.q):
                f_post_stream[l] = self.compute_dtype(self.read_field(f_0, index, l))
                f_post_collision[l] = self.compute_dtype(self.read_field(f_1, index, l))
            return f_post_collision, f_post_stream

        if self.operation_sequence == LBMOperationSequence.STREAM_THEN_COLLIDE:
            return functional_stream_then_collide, None
        elif self.operation_sequence == LBMOperationSequence.COLLIDE_THEN_STREAM:
            return functional_collide_then_stream, None
        else:
            raise ValueError(f"Unknown operation sequence: {self.operation_sequence}")

    def _construct_neon(self):
        # Use the warp functional for the NEON backend
        functional, _ = self._construct_warp()
        return functional, None


class MomentumTransfer(Operator):
    """
    An opertor for the momentum exchange method to compute the boundary force vector exerted on the solid geometry
    based on [1] as described in [3]. Ref [2] shows how [1] is applicable to curved geometries only by using a
    bounce-back method (e.g. Bouzidi) that accounts for curved boundaries.
    NOTE: this function should be called after BC's are imposed.
    [1] A.J.C. Ladd, Numerical simulations of particular suspensions via a discretized Boltzmann equation.
        Part 2 (numerical results), J. Fluid Mech. 271 (1994) 311-339.
    [2] R. Mei, D. Yu, W. Shyy, L.-S. Luo, Force evaluation in the lattice Boltzmann method involving
        curved geometry, Phys. Rev. E 65 (2002) 041203.
    [3] Caiazzo, A., & Junk, M. (2008). Boundary forces in lattice Boltzmann: Analysis of momentum exchange
        algorithm. Computers & Mathematics with Applications, 55(7), 1415-1423.

    Notes
    -----
    This method computes the force exerted on the solid geometry at each boundary node using the momentum exchange method.
    The force is computed based on the post-streaming and post-collision distribution functions. This method
    should be called after the boundary conditions are imposed.
    """

    def __init__(
        self,
        no_slip_bc_instance,
        operation_sequence: LBMOperationSequence = LBMOperationSequence.STREAM_THEN_COLLIDE,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        # Assign the no-slip boundary condition instance
        self.no_slip_bc_instance = no_slip_bc_instance
        self.operation_sequence = operation_sequence

        # Define the needed for the momentum transfer
        self.fetcher = FetchPopulations(
            no_slip_bc_instance=self.no_slip_bc_instance,
            operation_sequence=self.operation_sequence,
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )

        # Call the parent constructor
        super().__init__(
            velocity_set,
            precision_policy,
            compute_backend,
        )

        if self.compute_backend != ComputeBackend.JAX:
            # Allocate the force vector (the total integral value will be computed)
            _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
            self.force = wp.zeros((1), dtype=_u_vec)

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_0, f_1, bc_mask, missing_mask):
        """
        Parameters
        ----------
        f_0 : jax.numpy.ndarray
            The post-collision distribution function at each node in the grid.
        f_1 : jax.numpy.ndarray
            The buffer field the same size as f_0 (only given as input for consistency with the WARP backened API.)
        bc_mask : jax.numpy.ndarray
            A grid field with 0 everywhere except for boundary nodes which are designated
            by their respective boundary id's.
        missing_mask : jax.numpy.ndarray
            A grid field with lattice cardinality that specifies missing lattice directions
            for each boundary node.

        Returns
        -------
        jax.numpy.ndarray
            The force exerted on the solid geometry at each boundary node.
        """
        # Give the input post-collision populations, streaming once and apply the BC the find post-stream values.
        f_post_collision, f_post_stream = self.fetcher(f_0, f_1, bc_mask, missing_mask)

        # Compute momentum transfer
        boundary = bc_mask == self.no_slip_bc_instance.id
        new_shape = (self.velocity_set.q,) + boundary.shape[1:]
        boundary = lax.broadcast_in_dim(boundary, new_shape, tuple(range(self.velocity_set.d + 1)))

        # the following will return force as a grid-based field with zero everywhere except for boundary nodes.
        is_edge = jnp.logical_and(boundary, ~missing_mask[0])
        opp = self.velocity_set.opp_indices
        phi = f_post_collision[opp] + f_post_stream
        phi = jnp.where(jnp.logical_and(missing_mask, is_edge), phi, 0.0)
        force = jnp.tensordot(self.velocity_set.c[:, opp], phi, axes=(-1, 0))
        force_net = jnp.sum(force, axis=(i + 1 for i in range(self.velocity_set.d)))
        return force_net

    def _construct_warp(self):
        # Set local constants
        _c = self.velocity_set.c
        _opp_indices = self.velocity_set.opp_indices
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)
        _no_slip_id = self.no_slip_bc_instance.id

        # Find velocity index for (0, 0, 0)
        lattice_central_index = self.velocity_set.center_index

        @wp.func
        def functional(
            index: Any,
            f_0: Any,
            f_1: Any,
            bc_mask: Any,
            missing_mask: Any,
            force: Any,
        ):
            # Get the boundary id
            _boundary_id = self.read_field(bc_mask, index, 0)
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                _missing_mask[l] = self.read_field(missing_mask, index, l)

            # Determin if boundary is an edge by checking if center is missing
            is_edge = wp.bool(False)
            if _boundary_id == wp.uint8(_no_slip_id):
                if _missing_mask[lattice_central_index] == wp.uint8(0):
                    is_edge = wp.bool(True)

            # If the boundary is an edge then add the momentum transfer
            m = _u_vec()
            if is_edge:
                # fetch the post-collision and post-streaming populations
                f_post_collision, f_post_stream = self.fetcher_functional(index, f_0, f_1, _missing_mask)

                # Compute the momentum transfer
                for d in range(self.velocity_set.d):
                    m[d] = self.compute_dtype(0.0)
                    for l in range(self.velocity_set.q):
                        if _missing_mask[l] == wp.uint8(1):
                            phi = f_post_collision[_opp_indices[l]] + f_post_stream[l]
                            if _c[d, _opp_indices[l]] == 1:
                                m[d] += phi
                            elif _c[d, _opp_indices[l]] == -1:
                                m[d] -= phi
            # Atomic sum to get the total force vector
            wp.atomic_add(force, 0, m)

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f_0: wp.array4d(dtype=Any),
            f_1: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.uint8),
            force: wp.array(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Call the functional to compute the force
            functional(
                index,
                f_0,
                f_1,
                bc_mask,
                missing_mask,
                force,
            )

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_0, f_1, bc_mask, missing_mask):
        # Ensure the force is initialized to zero
        self.force *= self.compute_dtype(0.0)

        # Define the warp functionals needed for this operation
        self.fetcher_functional = self.fetcher.warp_functional

        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_0, f_1, bc_mask, missing_mask, self.force],
            dim=f_0.shape[1:],
        )
        return self.force.numpy()[0]

    def _construct_neon(self):
        import neon

        # Use the warp functional for the NEON backend
        functional, _ = self._construct_warp()

        @neon.Container.factory(name="MomentumTransfer")
        def container(
            f_0: Any,
            f_1: Any,
            bc_mask: Any,
            missing_mask: Any,
            force: Any,
        ):
            def container_launcher(loader: neon.Loader):
                loader.set_grid(bc_mask.get_grid())
                bc_mask_pn = loader.get_write_handle(bc_mask)
                missing_mask_pn = loader.get_write_handle(missing_mask)
                f_0_pn = loader.get_write_handle(f_0)
                f_1_pn = loader.get_write_handle(f_1)

                @wp.func
                def container_kernel(index: Any):
                    # apply the functional
                    functional(
                        index,
                        f_0_pn,
                        f_1_pn,
                        bc_mask_pn,
                        missing_mask_pn,
                        force,
                    )

                loader.declare_kernel(container_kernel)

            return container_launcher

        return functional, container

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(
        self,
        f_0,
        f_1,
        bc_mask,
        missing_mask,
        stream=0,
    ):
        import neon

        # Ensure the force is initialized to zero
        self.force *= self.compute_dtype(0.0)

        # Define the neon functionals needed for this operation
        self.fetcher_functional = self.fetcher.neon_functional

        # Launch the neon container
        c = self.neon_container(f_0, f_1, bc_mask, missing_mask, self.force)
        c.run(stream, container_runtime=neon.Container.ContainerRuntime.neon)
        return self.force.numpy()[0]

from functools import partial
import jax.numpy as jnp
from jax import jit, lax
import warp as wp
from typing import Any

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.stream import Stream


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
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        self.no_slip_bc_instance = no_slip_bc_instance
        self.stream = Stream(velocity_set, precision_policy, compute_backend)

        # Call the parent constructor
        super().__init__(
            velocity_set,
            precision_policy,
            compute_backend,
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f, bc_mask, missing_mask):
        """
        Parameters
        ----------
        f : jax.numpy.ndarray
            The post-collision distribution function at each node in the grid.
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
        f_post_collision = f
        f_post_stream = self.stream(f_post_collision)
        f_post_stream = self.no_slip_bc_instance(f_post_collision, f_post_stream, bc_mask, missing_mask)

        # Compute momentum transfer
        boundary = bc_mask == self.no_slip_bc_instance.id
        new_shape = (self.velocity_set.q,) + boundary.shape[1:]
        boundary = lax.broadcast_in_dim(boundary, new_shape, tuple(range(self.velocity_set.d + 1)))

        # the following will return force as a grid-based field with zero everywhere except for boundary nodes.
        opp = self.velocity_set.opp_indices
        phi = f_post_collision[opp] + f_post_stream
        phi = jnp.where(jnp.logical_and(boundary, missing_mask), phi, 0.0)
        force = jnp.tensordot(self.velocity_set.c[:, opp], phi, axes=(-1, 0))
        return force

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _c = self.velocity_set.c
        _opp_indices = self.velocity_set.opp_indices
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _missing_mask_vec = wp.vec(self.velocity_set.q, dtype=wp.uint8)  # TODO fix vec bool
        _no_slip_id = self.no_slip_bc_instance.id

        # Find velocity index for 0, 0, 0
        for l in range(self.velocity_set.q):
            if _c[0, l] == 0 and _c[1, l] == 0 and _c[2, l] == 0:
                zero_index = l
        _zero_index = wp.int32(zero_index)

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f_0: wp.array4d(dtype=Any),
            f_1: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
            force: wp.array(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Get the boundary id
            _boundary_id = bc_mask[0, index[0], index[1], index[2]]
            _missing_mask = _missing_mask_vec()
            for l in range(self.velocity_set.q):
                # TODO fix vec bool
                if missing_mask[l, index[0], index[1], index[2]]:
                    _missing_mask[l] = wp.uint8(1)
                else:
                    _missing_mask[l] = wp.uint8(0)

            # Determin if boundary is an edge by checking if center is missing
            is_edge = wp.bool(False)
            if _boundary_id == wp.uint8(_no_slip_id):
                if _missing_mask[_zero_index] == wp.uint8(0):
                    is_edge = wp.bool(True)

            # If the boundary is an edge then add the momentum transfer
            m = _u_vec()
            if is_edge:
                # Get the distribution function
                f_post_collision = _f_vec()
                for l in range(self.velocity_set.q):
                    f_post_collision[l] = f_0[l, index[0], index[1], index[2]]

                # Apply streaming (pull method)
                timestep = 0
                f_post_stream = self.stream.warp_functional(f_0, index)
                f_post_stream = self.no_slip_bc_instance.warp_functional(index, timestep, _missing_mask, f_0, f_1, f_post_collision, f_post_stream)

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

            wp.atomic_add(force, 0, m)

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_0, f_1, bc_mask, missing_mask):
        # Allocate the force vector (the total integral value will be computed)
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        force = wp.zeros((1), dtype=_u_vec)

        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_0, f_1, bc_mask, missing_mask, force],
            dim=f_0.shape[1:],
        )
        return force.numpy()[0]

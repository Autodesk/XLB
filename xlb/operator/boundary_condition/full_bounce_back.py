"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit, device_count
import jax.lax as lax
from functools import partial
import numpy as np
import warp as wp

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.boundary_condition import (
    BoundaryCondition,
    ImplementationStep,
)
from xlb.operator.boundary_condition.boundary_masker import (
    BoundaryMasker,
    IndicesBoundaryMasker,
)
from xlb.operator.boundary_condition.boundary_condition_registry import boundary_condition_registry


class FullBounceBack(BoundaryCondition):
    """
    Full Bounce-back boundary condition for a lattice Boltzmann method simulation.
    """
    id = boundary_condition_registry.register_boundary_condition(__qualname__)

    def __init__(
        self,
        boundary_masker: BoundaryMasker,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend,
    ):
        super().__init__(
            ImplementationStep.COLLISION,
            boundary_masker,
            velocity_set,
            precision_policy,
            compute_backend,
        )

    @classmethod
    def from_indices(
        cls, indices: np.ndarray, velocity_set, precision_policy, compute_backend
    ):
        """
        Create a full bounce-back boundary condition from indices.
        """
        # Create boundary mask
        boundary_mask = IndicesBoundaryMasker(
            indices, False, velocity_set, precision_policy, compute_backend
        )

        # Create boundary condition
        return cls(
            boundary_mask,
            velocity_set,
            precision_policy,
            compute_backend,
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0), donate_argnums=(1, 2, 3, 4))
    def apply_jax(self, f_pre, f_post, boundary_id, mask):
        boundary = boundary_id == self.id
        flip = jnp.repeat(boundary, self.velocity_set.q, axis=0)
        flipped_f = lax.select(flip, f_pre[self.velocity_set.opp_indices, ...], f_post)
        return flipped_f

    def _construct_warp(self):
        # Make constants for warp
        _opp_indices = wp.constant(self._warp_int_lattice_vec(self.velocity_set.opp_indices))
        _q = wp.constant(self.velocity_set.q)
        _d = wp.constant(self.velocity_set.d)

        # Construct the funcional to get streamed indices
        @wp.func
        def functional(
            f_pre: self._warp_lattice_vec,
            f_post: self._warp_lattice_vec,
            mask: self._warp_bool_lattice_vec,
        ):
            fliped_f = self._warp_lattice_vec()
            for l in range(_q):
                fliped_f[l] = f_pre[_opp_indices[l]]
            return fliped_f

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f_pre: self._warp_array_type,
            f_post: self._warp_array_type,
            f: self._warp_array_type,
            boundary: self._warp_bool_array_type,
            mask: self._warp_bool_array_type,
        ):
            # Get the global index
            i, j, k = wp.tid()

            # Make vectors for the lattice
            _f_pre = self._warp_lattice_vec()
            _f_post = self._warp_lattice_vec()
            _mask = self._warp_bool_lattice_vec()
            for l in range(_q):
                _f_pre[l] = f_pre[l, i, j, k]
                _f_post[l] = f_post[l, i, j, k]

                # TODO fix vec bool
                if mask[l, i, j, k]:
                    _mask[l] = wp.uint8(1)
                else:
                    _mask[l] = wp.uint8(0)

            # Check if the boundary is active
            if boundary[i, j, k]:
                _f = functional(_f_pre, _f_post, _mask)
            else:
                _f = _f_post

            # Write the result to the output
            for l in range(_q):
                f[l, i, j, k] = _f[l]

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, f, boundary, mask):
        # Launch the warp kernel
        wp.launch(
            self._kernel, inputs=[f_pre, f_post, f, boundary, mask], dim=f_pre.shape[1:]
        )
        return f

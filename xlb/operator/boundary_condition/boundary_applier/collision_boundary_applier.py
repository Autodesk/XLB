"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit, device_count
from functools import partial
import numpy as np
from enum import Enum

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator

# Import all collision boundary conditions
from xlb.boundary_condition.full_bounce_back import FullBounceBack


class CollisionBoundaryApplier(Operator):
    """
    Class for combining collision and boundary conditions together
    into a single operator.
    """

    def __init__(
        self,
        boundary_appliers: list[BoundaryApplier],
    ):
        # Set boundary conditions
        self.boundary_appliers = boundary_appliers

        # Check that all boundary conditions have the same implementation step other properties
        for bc in self.boundary_appliers:
            assert bc.implementation_step == ImplementationStep.COLLISION, (
                "All boundary conditions must be applied during the collision step."
            )

        # Get velocity set, precision policy, and compute backend
        velocity_sets = set([bc.velocity_set for bc in self.boundary_appliers])
        assert len(velocity_sets) == 1, "All velocity sets must be the same"
        velocity_set = velocity_sets.pop()
        precision_policies = set([bc.precision_policy for bc in self.boundary_appliers])
        assert len(precision_policies) == 1, "All precision policies must be the same"
        precision_policy = precision_policies.pop()
        compute_backends = set([bc.compute_backend for bc in self.boundary_appliers])
        assert len(compute_backends) == 1, "All compute backends must be the same"
        compute_backend = compute_backends.pop()

        # Make all possible collision boundary conditions to obtain the warp functions
        self.full_bounce_back = FullBounceBack(
            None, velocity_set, precision_policy, compute_backend
        )

        super().__init__(
            velocity_set,
            precision_policy,
            compute_backend,
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_pre, f_post, mask, boundary_id):
        """
        Apply collision boundary conditions
        """
        for bc in self.boundary_conditions:
            f_post, mask = bc.jax_implementation(f_pre, f_post, mask, boundary_id)
        return f_post, mask

    def _construct_warp(self):
        """
        Construct the warp kernel for the collision boundary condition.
        """

        # Make constants for warp
        _q = wp.constant(self.velocity_set.q)
        _d = wp.constant(self.velocity_set.d)

        # Get boolean constants for all boundary conditions
        if any([isinstance(bc, FullBounceBack) for bc in self.boundary_conditions]):
            _use_full_bounce_back = wp.constant(True)

        # Construct the funcional for all boundary conditions
        @wp.func
        def functional(
            f_pre: self._warp_lattice_vec,
            f_post: self._warp_lattice_vec,
            boundary_id: wp.uint8,
            mask: self._warp_bool_lattice_vec,
        ):
            # Apply all boundary conditions
            # Full bounce-back
            if _use_full_bounce_back:
                if boundary_id == self.full_bounce_back.id:
                    f_post = self.full_bounce_back.warp_functional(f_pre, f_post, mask)

            return f_post

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f_pre: self._warp_array_type,
            f_post: self._warp_array_type,
            f: self._warp_array_type,
            boundary_id: self._warp_uint8_array_type,
            mask: self._warp_bool_array_type,
        ):
            # Get the global index
            i, j, k = wp.tid()

            # Make vectors for the lattice
            _f_pre = self._warp_lattice_vec()
            _f_post = self._warp_lattice_vec()
            _mask = self._warp_bool_lattice_vec()
            _boundary_id = wp.uint8(boundary_id[0, i, j, k])
            for l in range(_q):
                _f_pre[l] = f_pre[l, i, j, k]
                _f_post[l] = f_post[l, i, j, k]

                # TODO fix vec bool
                if mask[l, i, j, k]:
                    _mask[l] = wp.uint8(1)
                else:
                    _mask[l] = wp.uint8(0)

            # Apply all boundary conditions
            if _boundary_id != wp.uint8(0):
                _f_post = functional(_f_pre, _f_post, _boundary_id, _mask)

            # Write the result to the output
            for l in range(_q):
                f[l, i, j, k] = _f_post[l]

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, f, boundary_id, mask):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel, inputs=[f_pre, f_post, f, boundary_id, mask], dim=f_pre.shape[1:]
        )
        return f

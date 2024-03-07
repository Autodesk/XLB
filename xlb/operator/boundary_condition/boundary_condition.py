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
from xlb.operator.boundary_condition.boundary_masker import (
    BoundaryMasker,
    IndicesBoundaryMasker,
)


# Enum for implementation step
class ImplementationStep(Enum):
    COLLISION = 1
    STREAMING = 2


class BoundaryCondition(Operator):
    """
    Base class for boundary conditions in a LBM simulation.
    """

    def __init__(
        self,
        implementation_step: ImplementationStep,
        boundary_masker: BoundaryMasker,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend.JAX,
    ):
        super().__init__(velocity_set, precision_policy, compute_backend)

        # Set implementation step
        self.implementation_step = implementation_step

        # Set boundary masker
        self.boundary_masker = boundary_masker

    @classmethod
    def from_function(
        cls,
        implementation_step: ImplementationStep,
        boundary_function,
        velocity_set,
        precision_policy,
        compute_backend,
    ):
        """
        Create a boundary condition from a function.
        """
        # Create boundary mask
        boundary_mask = BoundaryMasker.from_function(
            boundary_function, velocity_set, precision_policy, compute_backend
        )

        # Create boundary condition
        return cls(
            implementation_step,
            boundary_mask,
            velocity_set,
            precision_policy,
            compute_backend,
        )

    @classmethod
    def from_indices(
        cls,
        implementation_step: ImplementationStep,
        indices: np.ndarray,
        stream_indices: bool,
        velocity_set,
        precision_policy,
        compute_backend,
    ):
        """
        Create a boundary condition from indices and boundary id.
        """
        # Create boundary mask
        boundary_mask = IndicesBoundaryMasker(
            indices, stream_indices, velocity_set, precision_policy, compute_backend
        )

        # Create boundary condition
        return cls(
            implementation_step,
            boundary_mask,
            velocity_set,
            precision_policy,
            compute_backend,
        )

    @classmethod
    def from_stl(
        cls,
        implementation_step: ImplementationStep,
        stl_file: str,
        stream_indices: bool,
        velocity_set,
        precision_policy,
        compute_backend,
    ):
        """
        Create a boundary condition from an STL file.
        """
        raise NotImplementedError


class CollisionBoundaryCondition(Operator):
    """
    Class for combining collision and boundary conditions together
    into a single operator.
    """

    def __init__(
        self,
        boundary_conditions: list[BoundaryCondition],
    ):
        # Set boundary conditions
        self.boundary_conditions = boundary_conditions

        # Check that all boundary conditions have the same implementation step other properties
        for bc in self.boundary_conditions:
            assert bc.implementation_step == ImplementationStep.COLLISION, (
                "All boundary conditions must be applied during the collision step."
            )

        # Get velocity set, precision policy, and compute backend
        velocity_sets = set([bc.velocity_set for bc in self.boundary_conditions])
        assert len(velocity_sets) == 1, "All velocity sets must be the same"
        velocity_set = velocity_sets.pop()
        precision_policies = set([bc.precision_policy for bc in self.boundary_conditions])
        assert len(precision_policies) == 1, "All precision policies must be the same"
        precision_policy = precision_policies.pop()
        compute_backends = set([bc.compute_backend for bc in self.boundary_conditions])
        assert len(compute_backends) == 1, "All compute backends must be the same"
        compute_backend = compute_backends.pop()

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

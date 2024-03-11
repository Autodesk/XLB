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

    def __init__(
        self,
        boundary_masker: BoundaryMasker,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend,
    ):

        boundary_applier = FullBounceBackApplier(
            velocity_set, precision_policy, compute_backend
        )

        super().__init__(
            boundary_applier,
            boundary_masker,
            velocity_set,
            precision_policy,
            compute_backend,
        )

    @classmethod
    def from_indices(
        cls, velocity_set, precision_policy, compute_backend
    ):
        """
        Create a full bounce-back boundary condition from indices.
        """
        # Create boundary mask
        boundary_mask = IndicesBoundaryMasker(
            False, velocity_set, precision_policy, compute_backend
        )

        # Create boundary condition
        return cls(
            boundary_mask,
            velocity_set,
            precision_policy,
            compute_backend,
        )

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
from xlb.operator.boundary_condition.boundary_masker import BoundaryMasker
from xlb.operator.boundary_condition.boundary_applier import BoundaryApplier


# Enum for implementation step
class ImplementationStep(Enum):
    COLLISION = 1
    STREAMING = 2


class BoundaryCondition():
    """
    Base class for boundary conditions in a LBM simulation.

    Boundary conditions are unique in that they are not operators themselves,
    but rather hold operators for applying and making masks for boundary conditions.
    """

    def __init__(
        self,
        boundary_applier: BoundaryApplier,
        boundary_masker: BoundaryMasker,
    ):
        super().__init__(velocity_set, precision_policy, compute_backend)

        # Set operators
        self.boundary_applier = boundary_applier
        self.boundary_masker = boundary_masker

        # Get velocity set, precision policy, and compute backend
        velocity_sets = set([boundary_applier.velocity_set, boundary_masker.velocity_set])
        assert len(velocity_sets) == 1, "All velocity sets must be the same"
        velocity_set = velocity_sets.pop()
        precision_policies = set([boundary_applier.precision_policy, boundary_masker.precision_policy])
        assert len(precision_policies) == 1, "All precision policies must be the same"
        precision_policy = precision_policies.pop()
        compute_backends = set([boundary_applier.compute_backend, boundary_masker.compute_backend])
        assert len(compute_backends) == 1, "All compute backends must be the same"
        compute_backend = compute_backends.pop()

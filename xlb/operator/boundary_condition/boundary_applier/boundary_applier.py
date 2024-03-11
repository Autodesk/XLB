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

class BoundaryApplier(Operator):
    """
    Base class for boundary conditions in a LBM simulation.
    """

    def __init__(
        self,
        implementation_step: ImplementationStep,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend.JAX,
    ):
        super().__init__(velocity_set, precision_policy, compute_backend)

        # Set implementation step
        self.implementation_step = implementation_step

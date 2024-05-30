"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit, device_count
from functools import partial
import numpy as np
from enum import Enum, auto

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb import DefaultConfig

# Enum for implementation step
class ImplementationStep(Enum):
    COLLISION = auto()
    STREAMING = auto()


class BoundaryCondition(Operator):
    """
    Base class for boundary conditions in a LBM simulation.
    """

    def __init__(
        self,
        implementation_step: ImplementationStep,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        velocity_set = velocity_set or DefaultConfig.velocity_set
        precision_policy = precision_policy or DefaultConfig.default_precision_policy
        compute_backend = compute_backend or DefaultConfig.default_backend

        super().__init__(velocity_set, precision_policy, compute_backend)

        # Set the implementation step
        self.implementation_step = implementation_step

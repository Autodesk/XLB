"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit, device_count
from functools import partial
import numpy as np
from enum import Enum

from xlb.operator.operator import Operator
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend


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
        boundary_mask = IndicesBoundaryMask(
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

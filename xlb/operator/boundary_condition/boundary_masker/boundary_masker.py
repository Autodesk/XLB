# Base class for all equilibriums

import jax.numpy as jnp
from jax import jit
import warp as wp

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class BoundaryMasker(Operator):
    """
    Operator for creating a boundary mask
    """

    @classmethod
    def from_jax_func(
        cls, jax_func, precision_policy: PrecisionPolicy, velocity_set: VelocitySet
    ):
        """
        Create a boundary masker from a jax function
        """
        raise NotImplementedError

    @classmethod
    def from_warp_func(
        cls, warp_func, precision_policy: PrecisionPolicy, velocity_set: VelocitySet
    ):
        """
        Create a boundary masker from a warp function
        """
        raise NotImplementedError

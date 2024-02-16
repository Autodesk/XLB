# Base class for all equilibriums

from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Tuple

from xlb.global_config import GlobalConfig
from xlb.velocity_set.velocity_set import VelocitySet
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

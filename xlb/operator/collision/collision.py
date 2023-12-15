"""
Base class for Collision operators
"""

import jax.numpy as jnp
from jax import jit
from functools import partial
import numba

from xlb.compute_backend import ComputeBackend
from xlb.velocity_set import VelocitySet
from xlb.operator import Operator


class Collision(Operator):
    """
    Base class for collision operators.

    This class defines the collision step for the Lattice Boltzmann Method.

    Parameters
    ----------
    omega : float
        Relaxation parameter for collision step. Default value is 0.6.
    shear : bool
        Flag to indicate whether the collision step requires the shear stress.
    """

    def __init__(
            self,
            omega: float,
            velocity_set: VelocitySet,
            compute_backend=ComputeBackend.JAX,
        ):
        super().__init__(velocity_set, compute_backend)
        self.omega = omega

    def apply_jax(self, f, feq, rho, u):
        """
        Jax implementation of collision step.
        """
        raise NotImplementedError("Child class must implement apply_jax.")

    def construct_numba(self, velocity_set: VelocitySet, dtype=numba.float32):
        """
        Construct numba implementation of collision step.
        """
        raise NotImplementedError("Child class must implement construct_numba.")

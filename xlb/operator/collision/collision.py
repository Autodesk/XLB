"""
Base class for Collision operators
"""
from xlb.compute_backends import ComputeBackends
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
        compute_backend=ComputeBackends.JAX,
    ):
        super().__init__(velocity_set, compute_backend)
        self.omega = omega

"""
Base class for Collision operators
"""

from xlb.velocity_set import VelocitySet
from xlb.operator import Operator


class Collision(Operator):
    """
    Base class for collision operators.

    This class defines the collision step for the Lattice Boltzmann Method.
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        super().__init__(velocity_set, precision_policy, compute_backend)

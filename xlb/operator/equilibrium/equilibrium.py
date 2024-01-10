# Base class for all equilibriums
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backends import ComputeBackends
from xlb.operator.operator import Operator


class Equilibrium(Operator):
    """
    Base class for all equilibriums
    """

    def __init__(
        self,
        velocity_set: VelocitySet,
        compute_backend=ComputeBackends.JAX,
    ):
        super().__init__(velocity_set, compute_backend)

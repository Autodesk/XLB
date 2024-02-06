# Base class for all equilibriums
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.operator.operator import Operator


class Equilibrium(Operator):
    """
    Base class for all equilibriums
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        presision_policy=None,
        compute_backend=None,
    ):
        super().__init__(velocity_set, presision_policy, compute_backend)

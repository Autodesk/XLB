"""
Base class for Source operators
"""

from xlb.velocity_set import VelocitySet
from xlb.operator import Operator


class Source(Operator):
    """
    Base class for source operators.

    Source operators add a volumetric production/consumption term to the populations after
    collision. They are the scalar-transport counterpart of the momentum forcing operators
    in :mod:`xlb.operator.force`.
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        super().__init__(velocity_set, precision_policy, compute_backend)

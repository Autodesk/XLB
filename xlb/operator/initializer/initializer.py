from xlb.velocity_set import VelocitySet
from xlb.global_config import GlobalConfig
from xlb.compute_backends import ComputeBackends
from xlb.operator.operator import Operator
from xlb.grid.grid import Grid
import numpy as np
import jax


class Initializer(Operator):
    """
    Base class for all initializers.
    """

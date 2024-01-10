# Enum used to keep track of the compute backends

from enum import Enum

class ComputeBackends(Enum):
    JAX = 1
    WARP = 2

# Enum used to keep track of the compute backends

from enum import Enum

class ComputeBackend(Enum):
    JAX = 1
    NUMBA = 2
    PYTORCH = 3
    WARP = 4

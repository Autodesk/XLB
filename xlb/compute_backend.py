# Enum used to keep track of the compute backends

from enum import Enum, auto


class ComputeBackend(Enum):
    JAX = auto()
    WARP = auto()
    NEON = auto()

# Enum used to keep track of the compute backends

from enum import Enum, auto


class GridBackend(Enum):
    JAX = auto()
    WARP = auto()
    OOC = auto()

# Enum used to keep track of the compute backends

from enum import Enum, auto


class MeshVoxelizationMethod(Enum):
    AABB = auto()
    RAY = auto()
    AABB_FILL = auto()
    WINDING = auto()

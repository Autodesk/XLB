# Enum used to keep track of the physics types supported by different operators

from enum import Enum, auto


class PhysicsType(Enum):
    NSE = auto()  # Navier-Stokes Equations
    ADE = auto()  # Advection-Diffusion Equations

# Enum used to keep track of the physics types supported by different operators

from enum import Enum

class PhysicsType(Enum):
    NSE = 1  # Navier-Stokes Equations
    ADE = 2  # Advection-Diffusion Equations

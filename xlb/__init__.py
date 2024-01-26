# Enum classes
from xlb.compute_backends import ComputeBackends
from xlb.physics_type import PhysicsType


# Config
from .global_config import init


# Precision policy
import xlb.precision_policy

# Velocity Set
import xlb.velocity_set

# Operators
import xlb.operator.equilibrium
import xlb.operator.collision
import xlb.operator.stream
import xlb.operator.boundary_condition
# import xlb.operator.force
import xlb.operator.macroscopic

# Grids
import xlb.grid

# Solvers
import xlb.solver
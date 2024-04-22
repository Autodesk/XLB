# Enum classes
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy, Precision
from xlb.physics_type import PhysicsType

# Config
from .default_config import init, default_backend

# Velocity Set
import xlb.velocity_set

# Operators
import xlb.operator.equilibrium
import xlb.operator.collision
import xlb.operator.stream
import xlb.operator.boundary_condition
import xlb.operator.macroscopic

# Grids
import xlb.grid

# Solvers
import xlb.solver

# Utils
import xlb.utils

import xlb
from xlb.compute_backends import ComputeBackends
from xlb.precision_policy import Fp32Fp32

from xlb.solver import IncompressibleNavierStokes
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.stream import Stream
from xlb.global_config import GlobalConfig
from xlb.grid import Grid
from xlb.operator.initializer import EquilibriumInitializer, ConstInitializer

import numpy as np
import jax.numpy as jnp

xlb.init(precision_policy=Fp32Fp32, compute_backend=ComputeBackends.JAX, velocity_set=xlb.velocity_set.D2Q9)

grid_shape = (100, 100)
grid = Grid.create(grid_shape)

f_init = grid.create_field(cardinality=9, callback=EquilibriumInitializer(grid))

u_init = grid.create_field(cardinality=2, callback=ConstInitializer(grid, cardinality=2, const_value=0.0))   
rho_init = grid.create_field(cardinality=1, callback=ConstInitializer(grid, cardinality=1, const_value=1.0))


st = Stream(grid)

f_init = st(f_init)
print("here")
solver = IncompressibleNavierStokes(grid)

num_steps = 100
f = f_init
for step in range(num_steps):
    f = solver.step(f, timestep=step)
    print(f"Step {step+1}/{num_steps} complete")


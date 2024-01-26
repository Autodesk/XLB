import jax.numpy as jnp
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.stream import Stream
from xlb.velocity_set import D2Q9, D3Q27
from xlb.operator.collision.kbc import KBC, BGK
from xlb.compute_backends import ComputeBackends
from xlb.grid import Grid
import xlb


xlb.init(velocity_set=D2Q9(), compute_backend=ComputeBackends.JAX)

collision = BGK(omega=0.6)

# eq = QuadraticEquilibrium(velocity_set=D2Q9(), compute_backend=ComputeBackends.JAX)

# macro = Macroscopic(velocity_set=D2Q9(), compute_backend=ComputeBackends.JAX)

# s = Stream(velocity_set=D2Q9(), compute_backend=ComputeBackends.JAX)

Q = 19
# create random jnp arrays
f = jnp.ones((Q, 10, 10))
rho = jnp.ones((1, 10, 10))
u = jnp.zeros((2, 10, 10))
# feq = eq(rho, u)

print(collision(f, f))

grid = Grid.create(grid_shape=(10, 10), velocity_set=D2Q9(), compute_backend=ComputeBackends.JAX)

def advection_result(index):
    return 1.0


f = grid.initialize_pop(advection_result)

print(f)
print(f.sharding)
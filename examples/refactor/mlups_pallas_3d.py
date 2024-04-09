import xlb
import time
import argparse
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import Fp32Fp32
from xlb.solver import IncompressibleNavierStokes
from xlb.grid import Grid
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.equilibrium import QuadraticEquilibrium
import numpy as np
import jax.numpy as jnp

# Command line argument parsing
parser = argparse.ArgumentParser(
    description="3D Lattice Boltzmann Method Simulation using XLB"
)
parser.add_argument("cube_edge", type=int, help="Length of the edge of the cubic grid")
parser.add_argument(
    "num_steps", type=int, help="Number of timesteps for the simulation"
)
args = parser.parse_args()

# Initialize XLB
xlb.init(
    precision_policy=Fp32Fp32,
    compute_backend=ComputeBackend.PALLAS,
    velocity_set=xlb.velocity_set.D3Q19,
)

# Grid initialization
grid_shape = (args.cube_edge, args.cube_edge, args.cube_edge)
grid = Grid.create(grid_shape)


def initializer():
    rho = grid.create_field(cardinality=1) + 1.0
    u = grid.create_field(cardinality=3)

    sphere_center = np.array([s // 2 for s in grid_shape])
    sphere_radius = 10

    x, y, z = np.meshgrid(
        np.arange(grid_shape[0]),
        np.arange(grid_shape[1]),
        np.arange(grid_shape[2]),
        indexing="ij",
    )

    squared_dist = (
        (x - sphere_center[0]) ** 2
        + (y - sphere_center[1]) ** 2
        + (z - sphere_center[2]) ** 2
    )

    inside_sphere = squared_dist <= sphere_radius**2

    rho = jnp.where(inside_sphere, rho.at[0, x, y, z].add(0.001), rho)

    func_eq = QuadraticEquilibrium(compute_backend=ComputeBackend.JAX)
    f_eq = func_eq(rho, u)

    return f_eq


f = initializer()

solver = IncompressibleNavierStokes(grid, omega=1.0)

# AoT compile
f = solver.step(f, timestep=0)

# Start the simulation
start_time = time.time()

for step in range(args.num_steps):
    f = solver.step(f, timestep=step)

end_time = time.time()

# MLUPS calculation
total_lattice_updates = args.cube_edge**3 * args.num_steps
total_time_seconds = end_time - start_time
mlups = (total_lattice_updates / total_time_seconds) / 1e6
print(f"MLUPS: {mlups}")

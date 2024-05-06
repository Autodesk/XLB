import xlb
import time
import jax
import argparse
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import Fp32Fp32
from xlb.operator.initializer import EquilibriumInitializer

from xlb.solver import IncompressibleNavierStokes
from xlb.grid import grid_factory

parser = argparse.ArgumentParser(
    description="MLUPS for 3D Lattice Boltzmann Method Simulation (BGK)"
)
parser.add_argument("cube_edge", type=int, help="Length of the edge of the cubic grid")
parser.add_argument("num_steps", type=int, help="Timestep for the simulation")

args = parser.parse_args()

cube_edge = args.cube_edge
num_steps = args.num_steps


xlb.init(
    precision_policy=Fp32Fp32,
    compute_backend=ComputeBackend.PALLAS,
    velocity_set=xlb.velocity_set.D3Q19,
)

grid_shape = (cube_edge, cube_edge, cube_edge)
grid = Grid.create(grid_shape)

f = grid.create_field(cardinality=19)

print("f shape", f.shape)

solver = IncompressibleNavierStokes(grid, omega=1.0)

# Ahead-of-Time Compilation to remove JIT overhead
f = solver.step(f, timestep=0)

start_time = time.time()

for step in range(num_steps):
    f = solver.step(f, timestep=step)

end_time = time.time()
total_lattice_updates = cube_edge**3 * num_steps
total_time_seconds = end_time - start_time
mlups = (total_lattice_updates / total_time_seconds) / 1e6
print(f"MLUPS: {mlups}")

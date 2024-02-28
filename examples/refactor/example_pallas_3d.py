import xlb
from xlb.compute_backends import ComputeBackends
from xlb.precision_policy import Fp32Fp32

from xlb.solver import IncompressibleNavierStokes
from xlb.grid import Grid
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.utils import save_fields_vtk, save_image
import numpy as np
import jax.numpy as jnp

# Initialize XLB with Pallas backend for 3D simulation
xlb.init(
    precision_policy=Fp32Fp32,
    compute_backend=ComputeBackends.PALLAS,  # Changed to Pallas backend
    velocity_set=xlb.velocity_set.D3Q19,     # Changed to D3Q19 for 3D
)

grid_shape = (128, 128, 128)  # Adjusted for 3D grid
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

    func_eq = QuadraticEquilibrium(compute_backend=ComputeBackends.JAX)
    f_eq = func_eq(rho, u)

    return f_eq


f = initializer()

compute_macro = Macroscopic(compute_backend=ComputeBackends.JAX)

solver = IncompressibleNavierStokes(grid, omega=1.0)

def perform_io(f, step):
    rho, u = compute_macro(f)
    fields = {"rho": rho[0], "u_x": u[0], "u_y": u[1], "u_z": u[2]}
    save_fields_vtk(fields, step)
    # save_image function might not be suitable for 3D, consider alternative visualization
    print(f"Step {step + 1} complete")

num_steps = 1000
io_rate = 100
for step in range(num_steps):
    f = solver.step(f, timestep=step)

    if step % io_rate == 0:
        perform_io(f, step)

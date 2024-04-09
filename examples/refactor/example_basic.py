import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import Fp32Fp32

from xlb.solver import IncompressibleNavierStokes
from xlb.grid import Grid
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.utils import save_fields_vtk, save_image

xlb.init(
    precision_policy=Fp32Fp32,
    compute_backend=ComputeBackend.JAX,
    velocity_set=xlb.velocity_set.D2Q9,
)

grid_shape = (1000, 1000)
grid = Grid.create(grid_shape)


def initializer():
    rho = grid.create_field(cardinality=1) + 1.0
    u = grid.create_field(cardinality=2)

    circle_center = (grid_shape[0] // 2, grid_shape[1] // 2)
    circle_radius = 10

    for x in range(grid_shape[0]):
        for y in range(grid_shape[1]):
            if (x - circle_center[0]) ** 2 + (
                y - circle_center[1]
            ) ** 2 <= circle_radius**2:
                rho = rho.at[0, x, y].add(0.001)

    func_eq = QuadraticEquilibrium()
    f_eq = func_eq(rho, u)

    return f_eq


f = initializer()

compute_macro = Macroscopic()

solver = IncompressibleNavierStokes(grid, omega=1.0)


def perform_io(f, step):
    rho, u = compute_macro(f)
    fields = {"rho": rho[0], "u_x": u[0], "u_y": u[1]}
    save_fields_vtk(fields, step)
    save_image(rho[0], step)
    print(f"Step {step + 1} complete")


num_steps = 1000
io_rate = 100
for step in range(num_steps):
    f = solver.step(f, timestep=step)

    if step % io_rate == 0:
        perform_io(f, step)

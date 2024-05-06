import xlb
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.default_config import DefaultConfig
import warp as wp
from xlb.grid import grid_factory
from xlb.precision_policy import Precision
import xlb.velocity_set

xlb.init(
    default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
    default_backend=ComputeBackend.JAX,
    velocity_set=xlb.velocity_set.D3Q19,
)

grid_size = 50
grid_shape = (grid_size, grid_size, grid_size)
my_grid = grid_factory(grid_shape)
f = my_grid.create_field(cardinality=9)

# compute_macro = QuadraticEquilibrium()

# f_eq = compute_macro(rho, u)


# DefaultConfig.velocity_set.w




# def initializer():
#     rho = grid.create_field(cardinality=1) + 1.0
#     u = grid.create_field(cardinality=2)

#     circle_center = (grid_shape[0] // 2, grid_shape[1] // 2)
#     circle_radius = 10

#     for x in range(grid_shape[0]):
#         for y in range(grid_shape[1]):
#             if (x - circle_center[0]) ** 2 + (
#                 y - circle_center[1]
#             ) ** 2 <= circle_radius**2:
#                 rho = rho.at[0, x, y].add(0.001)

#     func_eq = QuadraticEquilibrium()
#     f_eq = func_eq(rho, u)

#     return f_eq



# solver = IncompressibleNavierStokes(grid, omega=1.0)


# def perform_io(f, step):
#     rho, u = compute_macro(f)
#     fields = {"rho": rho[0], "u_x": u[0], "u_y": u[1]}
#     save_fields_vtk(fields, step)
#     save_image(rho[0], step)
#     print(f"Step {step + 1} complete")


# num_steps = 1000
# io_rate = 100
# for step in range(num_steps):
#     f = solver.step(f, timestep=step)

#     if step % io_rate == 0:
#         perform_io(f, step)

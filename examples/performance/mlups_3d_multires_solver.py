import xlb
import argparse
import time
import warp as wp
import numpy as np

# add a directory to the PYTHON PATH
import sys
# sys.path.append('/home/max/repos/neon/warping/neon_warp_testing/neon_py_bindings/py/')
import neon

from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import multires_grid_factory
from xlb.operator.stepper import MultiresIncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import FullwayBounceBackBC, EquilibriumBC
from xlb.distribute import distribute

def parse_arguments():
    parser = argparse.ArgumentParser(description="MLUPS for 3D Lattice Boltzmann Method Simulation (BGK)")
    # Positional arguments
    parser.add_argument("cube_edge", type=int, help="Length of the edge of the cubic grid")
    parser.add_argument("num_steps", type=int, help="Timestep for the simulation")
    parser.add_argument("backend", type=str, help="Backend for the simulation (jax, warp or neon)")
    parser.add_argument("precision", type=str, help="Precision for the simulation (e.g., fp32/fp32)")

    # Optional arguments
    parser.add_argument("--num_devices", type=int, default=0, help="Number of devices for the simulation (default: 0)")
    parser.add_argument("--velocity_set", type=str, default='D3Q19',
                        help="Lattice type: D3Q19 or D3Q27 (default: D3Q19)"
                        )

    return parser.parse_args()


def setup_simulation(args):
    backend = None
    if args.backend == "jax": backend = ComputeBackend.JAX
    elif args.backend == "warp": backend = ComputeBackend.WARP
    elif args.backend == "neon": backend = ComputeBackend.NEON
    if backend is None:
        raise ValueError("Invalid backend")

    precision_policy_map = {
        "fp32/fp32": PrecisionPolicy.FP32FP32,
        "fp64/fp64": PrecisionPolicy.FP64FP64,
        "fp64/fp32": PrecisionPolicy.FP64FP32,
        "fp32/fp16": PrecisionPolicy.FP32FP16,
    }
    precision_policy = precision_policy_map.get(args.precision)
    if precision_policy is None:
        raise ValueError("Invalid precision")

    velocity_set = None
    if args.velocity_set == 'D3Q19': velocity_set = xlb.velocity_set.D3Q19(precision_policy=precision_policy, backend=backend)
    elif args.velocity_set == 'D3Q27': velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, backend=backend)
    if velocity_set is None:
        raise ValueError("Invalid velocity set")

    xlb.init(
        velocity_set=velocity_set,
        default_backend=backend,
        default_precision_policy=precision_policy,
    )

    return backend, precision_policy


def run(backend, precision_policy, grid_shape, num_steps):
    # Create grid and setup boundary conditions
    velocity_set = xlb.velocity_set.D3Q19(precision_policy=precision_policy, backend=backend)

    dim = neon.Index_3d(grid_shape[0],
                        grid_shape[1],
                        grid_shape[2])
    level_zero_mask = np.ones((dim.x//2, dim.y, dim.z), dtype=int)
    level_zero_mask = np.ascontiguousarray(level_zero_mask, dtype=np.int32)

    level_one_mask = np.ones((dim.x//2, dim.y, dim.z), dtype=int)
    level_one_mask = np.ascontiguousarray(level_one_mask, dtype=np.int32)

    #
    # level_one_mask = np.zeros((2, 2, 2), dtype=int)
    # level_one_mask[0, 0, 0] = 1
    # level_one_mask[0, 0, 1] = 0
    # level_one_mask[0, 1, 0] = 0
    # level_one_mask[1, 1, 1] = 1
    #
    # grid = neon.mGrid(bk, dim,
    #                   sparsity_pattern_list=[
    #                       np.ascontiguousarray(maskZero, dtype=np.int32),
    #                       np.ascontiguousarray(maskOne, dtype=np.int32),
    #                   ],
    #                   sparsity_pattern_origins=[neon.Index_3d(0, 0, 0),
    #                                             neon.Index_3d(0, 0, 0)],
    #                   stencil=[[0, 0, 0], [1, 0, 0]], )

    grid = multires_grid_factory(grid_shape, velocity_set=velocity_set,
                                 sparsity_pattern_list=[level_one_mask, level_zero_mask, ],
                                 sparsity_pattern_origins=[ neon.Index_3d(dim.x//2+1, 0, 0), neon.Index_3d(0, 0, 0),])

    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    lid = box_no_edge["top"]
    walls = [box["bottom"][i] + box["left"][i] + box["right"][i] + box["front"][i] + box["back"][i] for i in range(len(grid.shape))]
    walls = np.unique(np.array(walls), axis=-1).tolist()

    prescribed_vel = 0.05

    boundary_conditions = [EquilibriumBC(rho=1.0, u=(prescribed_vel, 0.0, 0.0), indices=lid), FullwayBounceBackBC(indices=walls)]

    # Create stepper
    stepper = MultiresIncompressibleNavierStokesStepper(grid=grid, boundary_conditions=boundary_conditions, collision_type="BGK")

    Re = 10000.0
    clength = grid_shape[0] - 1
    visc = prescribed_vel * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)

    # # Initialize fields and run simulation
    # omega = 1.0

    sim = xlb.helper.Nse_multires_simulation(grid, velocity_set, stepper, omega)
    print("start timing")
    start_time = time.time()

    for i in range(num_steps):
        print(f"step {i}")
        sim.step()
        if i%500 == 0:
            sim.export_macroscopic("u_lid_driven_cavity_")
    wp.synchronize()
    t = time.time() - start_time

    sim.export_macroscopic("u_lid_driven_cavity_")
    return t


def calculate_mlups(cube_edge, num_steps, elapsed_time):
    total_lattice_updates = cube_edge**3 * num_steps
    mlups = (total_lattice_updates / elapsed_time) / 1e6
    return mlups

def post_process(macro, rho, u, f_0,  i):
    # Write the results. We'll use JAX backend for the post-processing
    # import jax.numpy as jnp
    # if not isinstance(f_0, jnp.ndarray):
    #     # If the backend is warp, we need to drop the last dimension added by warp for 2D simulations
    #     f_0 = wp.to_jax(f_0)[..., 0]
    # else:
    #     f_0 = f_0
    rho, u = macro(f_0, rho, u )
    wp.synchronize()
    u.update_host(0)
    rho.update_host(0)
    wp.synchronize()
    u.export_vti(f"u_lid_driven_cavity_{i}.vti", 'u')
    rho.export_vti(f"rho_lid_driven_cavity_{i}.vti", 'rho')

    pass

    # # remove boundary cells
    # rho = rho[:, 1:-1, 1:-1, 1:-1]
    # u = u[:, 1:-1, 1:-1, 1:-1]
    # u_magnitude = (u[0] ** 2 + u[1] ** 2) ** 0.5
    #
    # fields = {"rho": rho[0], "u_x": u[0], "u_y": u[1], "u_magnitude": u_magnitude}
    #
    # # save_fields_vtk(fields, timestep=i, prefix="lid_driven_cavity")
    # ny=fields["u_magnitude"].shape[1]
    # from xlb.utils import  save_image
    # save_image(fields["u_magnitude"][:, ny//2, :], timestep=i, prefix="lid_driven_cavity")

def main():

    args = parse_arguments()
    backend, precision_policy = setup_simulation(args)
    grid_shape = (args.cube_edge, args.cube_edge, args.cube_edge)
    elapsed_time = run(backend, precision_policy, grid_shape, args.num_steps)
    mlups = calculate_mlups(args.cube_edge, args.num_steps, elapsed_time)

    print(f"Simulation completed in {elapsed_time:.2f} seconds")
    print(f"MLUPs: {mlups:.2f}")


if __name__ == "__main__":
    main()

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

    def peel(dim, idx, peel_level, outwards):
        if outwards:
            xIn =  idx.x <= peel_level or idx.x >= dim.x -1 -peel_level
            yIn =  idx.y <= peel_level or idx.y >= dim.y -1 -peel_level
            zIn =  idx.z <= peel_level or idx.z >= dim.z -1 - peel_level
            return xIn or yIn or zIn
        else:
            xIn = idx.x >= peel_level and idx.x <= dim.x - 1 - peel_level
            yIn = idx.y >= peel_level and idx.y <= dim.y - 1 - peel_level
            zIn = idx.z >= peel_level and idx.z <= dim.z - 1 - peel_level
            return xIn and yIn and zIn


    dim = neon.Index_3d(grid_shape[0],
                        grid_shape[1],
                        grid_shape[2])

    def get_peeled_np(level, width):
        divider = 2**level
        m = neon.Index_3d(dim.x // divider , dim.y // divider, dim.z // divider)
        if level == 0:
            m = dim

        mask = np.zeros((m.x, m.y, m.z), dtype=int)
        mask = np.ascontiguousarray(mask, dtype=np.int32)
        # loop over all the elements in mask and set to one any that have x=0 or y=0 or z=0
        for i in range(m.x):
            for j in range(m.y):
                for k in range(m.z):
                    idx = neon.Index_3d(i, j, k)
                    val = 0
                    if peel(m, idx, m.x / width, True):
                        val = 1
                    mask[i, j, k] = val
        return mask

    levels = []

    l0 = get_peeled_np(0, 17)
    l1 = get_peeled_np(1, 7)
    l2 = get_peeled_np(2, 4)

    num_levels = 4
    lastLevel = num_levels -1
    divider = 2**lastLevel
    m = neon.Index_3d(dim.x // divider +1, dim.y // divider+1, dim.z // divider+1)
    lastLevel = np.ones((m.x, m.y, m.z), dtype=int)
    lastLevel = np.ascontiguousarray(lastLevel, dtype=np.int32)

    levels = [l0, l1, l2, lastLevel]

    grid = multires_grid_factory(grid_shape, velocity_set=velocity_set,
                                 sparsity_pattern_list=levels,
                                 sparsity_pattern_origins=[ neon.Index_3d(0, 0, 0)]*len(levels),)

    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    lid = box_no_edge["top"]
    walls = [box["bottom"][i] + box["left"][i] + box["right"][i] + box["front"][i] + box["back"][i] for i in range(len(grid.shape))]
    walls = np.unique(np.array(walls), axis=-1).tolist()

    prescribed_vel = 0.1

    boundary_conditions = [EquilibriumBC(rho=1.0, u=(prescribed_vel, 0.0, 0.0), indices=lid),
                           EquilibriumBC(rho=1.0, u=(0.0, 0.0, 0.0), indices=walls)]

    # Create stepper
    stepper = MultiresIncompressibleNavierStokesStepper(grid=grid, boundary_conditions=boundary_conditions, collision_type="BGK")

    Re = 5000.0

    clength = grid_shape[0] - 1
    visc = prescribed_vel * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)
    #omega = 1.0


    # # Initialize fields and run simulation
    # omega = 1.0

    sim = xlb.helper.Nse_multires_simulation(grid, velocity_set, stepper, omega)

    # sim.export_macroscopic("Initial_")
    # sim.step()

    print("start timing")
    wp.synchronize()

    start_time = time.time()
    for i in range(num_steps):
        sim.step()
        if i%100 == 0:
            print(f"step {i}")
        #    sim.export_macroscopic("u_lid_driven_cavity_")
    wp.synchronize()
    t = time.time() - start_time
    print(f"Timing  {t}")

    sim.export_macroscopic("u_lid_driven_cavity_")
    return {"time":t, "num_levels":num_levels}


def calculate_mlups(cube_edge, num_steps, elapsed_time, num_levels):
    num_step_finer = num_steps * 2**(num_levels-1)
    total_lattice_updates = cube_edge**3 * num_step_finer
    mlups = (total_lattice_updates / elapsed_time) / 1e6
    return {"EMLUPS":mlups, "finer_steps":num_step_finer}


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
    stats = run(backend, precision_policy, grid_shape, args.num_steps)
    mlups_stats = calculate_mlups(args.cube_edge, args.num_steps, stats['time'], stats['num_levels'])

    print(f"Simulation completed in {stats['time']:.2f} seconds")
    print(f"Number of levels {stats['num_levels']}")
    print(f"Cube edge {args.cube_edge}")
    print(f"Coarse Iterations {args.num_steps}")
    finer_steps = mlups_stats["finer_steps"]
    print(f"Fine Iterations {finer_steps}")
    print(f"EMLUPs: {mlups_stats["EMLUPS"]:.2f}")


if __name__ == "__main__":
    main()

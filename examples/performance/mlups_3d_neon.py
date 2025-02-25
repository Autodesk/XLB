from warp.examples.fem.example_convection_diffusion import velocity

import xlb
import argparse
import time
import warp as wp
import numpy as np

# add a directory to the PYTHON PATH
import sys
sys.path.append('/home/max/repos/neon/warping/neon_warp_testing/neon_py_bindings/py/')
import neon

from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
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
    grid = grid_factory(grid_shape, velocity_set=velocity_set)
    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    lid = box_no_edge["top"]
    walls = [box["bottom"][i] + box["left"][i] + box["right"][i] + box["front"][i] + box["back"][i] for i in range(len(grid.shape))]
    walls = np.unique(np.array(walls), axis=-1).tolist()

    boundary_conditions = [EquilibriumBC(rho=1.0, u=(0.02, 0.0, 0.0), indices=lid), FullwayBounceBackBC(indices=walls)]

    # Create stepper
    stepper = IncompressibleNavierStokesStepper(grid=grid, boundary_conditions=boundary_conditions, collision_type="BGK")

    # Distribute if using JAX backend
    if backend == ComputeBackend.JAX:
        stepper = distribute(
            stepper,
            grid,
            velocity_set,
        )

    # Initialize fields and run simulation
    omega = 1.0
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()
    start_time = time.time()

    for i in range(num_steps):
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, i)
        f_0, f_1 = f_1, f_0
    wp.synchronize()

    return time.time() - start_time


def calculate_mlups(cube_edge, num_steps, elapsed_time):
    total_lattice_updates = cube_edge**3 * num_steps
    mlups = (total_lattice_updates / elapsed_time) / 1e6
    return mlups


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

import xlb
import argparse
import time
import warp as wp
import numpy as np
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import FullwayBounceBackBC, EquilibriumBC
from xlb.distribute import distribute

# -------------------------- Simulation Setup --------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(description="MLUPS for 3D Lattice Boltzmann Method Simulation (BGK)")
    parser.add_argument("cube_edge", type=int, help="Length of the edge of the cubic grid")
    parser.add_argument("num_steps", type=int, help="Number of timesteps for the simulation")
    parser.add_argument("compute_backend", type=str, help="Backend for the simulation (jax or warp)")
    parser.add_argument("precision", type=str, help="Precision for the simulation (e.g., fp32/fp32)")
    return parser.parse_args()


def setup_simulation(args):
    compute_backend = ComputeBackend.JAX if args.compute_backend == "jax" else ComputeBackend.WARP
    precision_policy_map = {
        "fp32/fp32": PrecisionPolicy.FP32FP32,
        "fp64/fp64": PrecisionPolicy.FP64FP64,
        "fp64/fp32": PrecisionPolicy.FP64FP32,
        "fp32/fp16": PrecisionPolicy.FP32FP16,
    }
    precision_policy = precision_policy_map.get(args.precision)
    if precision_policy is None:
        raise ValueError("Invalid precision specified.")

    xlb.init(
        velocity_set=xlb.velocity_set.D3Q19(precision_policy=precision_policy, compute_backend=compute_backend),
        default_backend=compute_backend,
        default_precision_policy=precision_policy,
    )
    return compute_backend, precision_policy


def run_simulation(compute_backend, precision_policy, grid_shape, num_steps):
    grid = grid_factory(grid_shape)
    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)

    lid = box_no_edge["top"]
    walls = [box["bottom"][i] + box["left"][i] + box["right"][i] + box["front"][i] + box["back"][i] for i in range(len(grid.shape))]
    walls = np.unique(np.array(walls), axis=-1).tolist()

    boundary_conditions = [
        EquilibriumBC(rho=1.0, u=(0.02, 0.0, 0.0), indices=lid),
        FullwayBounceBackBC(indices=walls),
    ]

    stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type="BGK",
    )

    # Distribute if using JAX
    if compute_backend == ComputeBackend.JAX:
        stepper = distribute(
            stepper,
            grid,
            xlb.velocity_set.D3Q19(precision_policy=precision_policy, compute_backend=compute_backend),
        )

    # Initialize fields
    omega = 1.0
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

    start_time = time.time()
    for i in range(num_steps):
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, i)
        f_0, f_1 = f_1, f_0
    wp.synchronize()
    elapsed_time = time.time() - start_time

    return elapsed_time


def calculate_mlups(cube_edge, num_steps, elapsed_time):
    total_lattice_updates = cube_edge**3 * num_steps
    mlups = (total_lattice_updates / elapsed_time) / 1e6
    return mlups


# -------------------------- Simulation Loop --------------------------

args = parse_arguments()
compute_backend, precision_policy = setup_simulation(args)
grid_shape = (args.cube_edge, args.cube_edge, args.cube_edge)

elapsed_time = run_simulation(compute_backend=compute_backend, precision_policy=precision_policy, grid_shape=grid_shape, num_steps=args.num_steps)

mlups = calculate_mlups(args.cube_edge, args.num_steps, elapsed_time)

print(f"Simulation completed in {elapsed_time:.2f} seconds")
print(f"MLUPs: {mlups:.2f}")

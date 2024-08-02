import xlb
import argparse
import time
import warp as wp
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.helper import create_nse_fields, initialize_eq
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import FullwayBounceBackBC, EquilibriumBC
from xlb.distribute import distribute


def parse_arguments():
    parser = argparse.ArgumentParser(description="MLUPS for 3D Lattice Boltzmann Method Simulation (BGK)")
    parser.add_argument("cube_edge", type=int, help="Length of the edge of the cubic grid")
    parser.add_argument("num_steps", type=int, help="Timestep for the simulation")
    parser.add_argument("backend", type=str, help="Backend for the simulation (jax or warp)")
    parser.add_argument("precision", type=str, help="Precision for the simulation (e.g., fp32/fp32)")
    return parser.parse_args()


def setup_simulation(args):
    backend = ComputeBackend.JAX if args.backend == "jax" else ComputeBackend.WARP
    precision_policy_map = {
        "fp32/fp32": PrecisionPolicy.FP32FP32,
        "fp64/fp64": PrecisionPolicy.FP64FP64,
        "fp64/fp32": PrecisionPolicy.FP64FP32,
        "fp32/fp16": PrecisionPolicy.FP32FP16,
    }
    precision_policy = precision_policy_map.get(args.precision)
    if precision_policy is None:
        raise ValueError("Invalid precision")

    xlb.init(
        velocity_set=xlb.velocity_set.D3Q19(),
        default_backend=backend,
        default_precision_policy=precision_policy,
    )

    return backend, precision_policy


def create_grid_and_fields(cube_edge):
    grid_shape = (cube_edge, cube_edge, cube_edge)
    grid, f_0, f_1, missing_mask, boundary_mask = create_nse_fields(grid_shape)

    return grid, f_0, f_1, missing_mask, boundary_mask


def define_boundary_indices(grid):
    lid = grid.boundingBoxIndices["top"]
    walls = [
        grid.boundingBoxIndices["bottom"][i]
        + grid.boundingBoxIndices["left"][i]
        + grid.boundingBoxIndices["right"][i]
        + grid.boundingBoxIndices["front"][i]
        + grid.boundingBoxIndices["back"][i]
        for i in range(xlb.velocity_set.D3Q19().d)
    ]
    return lid, walls


def setup_boundary_conditions(grid):
    lid, walls = define_boundary_indices(grid)
    bc_top = EquilibriumBC(rho=1.0, u=(0.02, 0.0, 0.0), indices=lid)
    bc_walls = FullwayBounceBackBC(indices=walls)
    return [bc_top, bc_walls]


def run(f_0, f_1, backend, grid, boundary_mask, missing_mask, num_steps):
    omega = 1.0
    stepper = IncompressibleNavierStokesStepper(omega, boundary_conditions=setup_boundary_conditions(grid))

    if backend == ComputeBackend.JAX:
        stepper = distribute(
            stepper,
            grid,
            xlb.velocity_set.D3Q19(),
        )

    start_time = time.time()

    for i in range(num_steps):
        f_1 = stepper(f_0, f_1, boundary_mask, missing_mask, i)
        f_0, f_1 = f_1, f_0
    wp.synchronize()

    end_time = time.time()
    return end_time - start_time


def calculate_mlups(cube_edge, num_steps, elapsed_time):
    total_lattice_updates = cube_edge**3 * num_steps
    mlups = (total_lattice_updates / elapsed_time) / 1e6
    return mlups


def main():
    args = parse_arguments()
    backend, precision_policy = setup_simulation(args)
    grid, f_0, f_1, missing_mask, boundary_mask = create_grid_and_fields(args.cube_edge)
    f_0 = initialize_eq(f_0, grid, xlb.velocity_set.D3Q19(), backend)

    elapsed_time = run(f_0, f_1, backend, grid, boundary_mask, missing_mask, args.num_steps)
    mlups = calculate_mlups(args.cube_edge, args.num_steps, elapsed_time)

    print(f"Simulation completed in {elapsed_time:.2f} seconds")
    print(f"MLUPs: {mlups:.2f}")


if __name__ == "__main__":
    main()

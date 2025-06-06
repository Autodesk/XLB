import xlb
import argparse
import time
import warp as wp
import numpy as np
import neon
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import multires_grid_factory
from xlb.operator.boundary_condition import FullwayBounceBackBC, EquilibriumBC


def parse_arguments():
    """Parse command line arguments for the MLUPS benchmark."""
    parser = argparse.ArgumentParser(
        description="MLUPS Benchmark for 3D Multires Lattice Boltzmann Method Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples (copy-paste ready):
  # Basic BGK test (uses D3Q19) - 2 levels, simple
  python examples/performance/mlups_3d_multires.py 128 50 neon fp32/fp32

  # KBC test (uses D3Q27) - 2 levels, more accurate collision model
  python examples/performance/mlups_3d_multires.py 128 50 neon fp32/fp32 --collision_type KBC

  # Nested refinement - 4 levels, complex grid structure
  python examples/performance/mlups_3d_multires.py 128 50 neon fp32/fp32 --levels 4

  # Verbose output to see all details
  python examples/performance/mlups_3d_multires.py 128 50 neon fp32/fp32 --verbose

  # Larger grid for performance testing
  python examples/performance/mlups_3d_multires.py 256 100 neon fp32/fp32 --collision_type KBC --levels 3
        """.strip(),
    )

    parser.add_argument("cube_edge", type=int, help="Length of the edge of the cubic grid")
    parser.add_argument("num_steps", type=int, help="Number of timesteps for the simulation")
    parser.add_argument("compute_backend", type=str, help="Backend for the simulation (only 'neon' supported)")
    parser.add_argument("precision", type=str, help="Precision for the simulation (e.g., fp32/fp32)")

    parser.add_argument("--num_devices", type=int, default=0, help="Number of devices")
    parser.add_argument("--levels", type=int, default=2, help="Number of refinement levels (2=simple, >2=nested)")
    parser.add_argument("--collision_type", type=str, default="BGK", choices=["BGK", "KBC"], help="Collision type (BGK uses D3Q19, KBC uses D3Q27)")
    parser.add_argument("--reynolds", type=float, default=5000.0, help="Reynolds number")
    parser.add_argument("--velocity", type=float, default=0.1, help="Prescribed velocity magnitude")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def get_velocity_set_for_collision(collision_type, precision_policy, compute_backend):
    """
    Automatically select the appropriate velocity set based on collision type.
    BGK uses D3Q19, KBC uses D3Q27 for better accuracy.
    """
    velocity_set_map = {
        "BGK": xlb.velocity_set.D3Q19,
        "KBC": xlb.velocity_set.D3Q27,
    }

    velocity_set_class = velocity_set_map[collision_type]
    return velocity_set_class(precision_policy=precision_policy, compute_backend=compute_backend)


def validate_arguments(args):
    """Validate command line arguments and provide helpful error messages."""
    if args.compute_backend != "neon":
        raise ValueError("Only 'neon' backend supports multi-resolution simulations!")

    if args.cube_edge < 128:
        raise ValueError("Cube edge must be at least 128 for meaningful multires simulation!")

    if args.num_steps < 1:
        raise ValueError("Number of steps must be positive!")

    if args.reynolds <= 0:
        raise ValueError("Reynolds number must be positive!")

    if args.velocity <= 0 or args.velocity >= 0.3:
        raise ValueError("Velocity must be positive and less than 0.3 for stability!")

    if args.levels < 2:
        raise ValueError("Number of levels must be at least 2!")


def setup_simulation(args):
    """Setup XLB simulation with specified parameters."""
    validate_arguments(args)

    compute_backend = ComputeBackend.NEON

    precision_policy_map = {
        "fp32/fp32": PrecisionPolicy.FP32FP32,
        "fp64/fp64": PrecisionPolicy.FP64FP64,
        "fp64/fp32": PrecisionPolicy.FP64FP32,
        "fp32/fp16": PrecisionPolicy.FP32FP16,
    }

    precision_policy = precision_policy_map.get(args.precision)
    if precision_policy is None:
        raise ValueError(f"Invalid precision: {args.precision}. Valid options: {list(precision_policy_map.keys())}")

    velocity_set = get_velocity_set_for_collision(args.collision_type, precision_policy, compute_backend)

    xlb.init(
        velocity_set=velocity_set,
        default_backend=compute_backend,
        default_precision_policy=precision_policy,
    )

    return velocity_set


def print_simulation_header(args, velocity_set):
    """Print a comprehensive simulation header with all parameters."""
    print("\n" + "=" * 80)
    print("XLB MULTIRES PERFORMANCE BENCHMARK")
    print("=" * 80)
    print("Configuration:")
    print(f"  Grid size: {args.cube_edge}³ = {args.cube_edge**3:,} cells")
    print(f"  Refinement levels: {args.levels} ({'simple' if args.levels == 2 else 'nested'})")
    print(f"  Collision model: {args.collision_type}")
    print(f"  Velocity set: {velocity_set.__class__.__name__} (q={velocity_set.q})")
    print(f"  Backend: {args.compute_backend}")
    print(f"  Precision: {args.precision}")
    print(f"  Reynolds number: {args.reynolds:,.0f}")
    print(f"  Prescribed velocity: {args.velocity}")
    print(f"  Timesteps: {args.num_steps:,}")
    print("=" * 80)


def create_multires_grid(grid_shape, velocity_set, num_levels=2, peel_width=8):
    """
    Create a multires grid. For num_levels=2, creates a simple 2-level grid.
    For num_levels>2, creates a nested refinement pattern where fine levels
    are nested inside coarser levels, with finest resolution at the center.
    """
    if num_levels == 2:
        return create_simple_2level_grid(grid_shape, velocity_set)
    else:
        return create_nested_grid(grid_shape, velocity_set, num_levels, peel_width)


def create_simple_2level_grid(grid_shape, velocity_set):
    """Create a simple 2-level grid for basic multires simulation."""
    coarse_level = np.ones((grid_shape[0] // 2, grid_shape[1] // 2, grid_shape[2] // 2), dtype=np.int32)

    fine_size = min(40, grid_shape[0] // 3)
    fine_level = np.ones((fine_size, fine_size, fine_size), dtype=np.int32)
    fine_level = np.ascontiguousarray(fine_level, dtype=np.int32)

    levels = [fine_level, coarse_level]

    fine_origin_offset = (grid_shape[0] - fine_size) // 2
    level_origins = [(fine_origin_offset, fine_origin_offset, fine_origin_offset), (0, 0, 0)]

    grid = multires_grid_factory(
        grid_shape,
        velocity_set=velocity_set,
        sparsity_pattern_list=levels,
        sparsity_pattern_origins=[neon.Index_3d(*origin) for origin in level_origins],
    )

    return grid, levels


def create_nested_grid(grid_shape, velocity_set, num_levels, peel_width):
    """
    Create a nested refinement grid where fine levels are nested inside coarser levels.
    This creates a pattern where the finest resolution is at the center, progressively
    getting coarser towards the boundaries.
    """

    def should_include_point(dim, idx, peel_level, outwards=True):
        """Determine if a point should be included based on peeling criteria."""
        if outwards:
            x_in = idx.x <= peel_level or idx.x >= dim.x - 1 - peel_level
            y_in = idx.y <= peel_level or idx.y >= dim.y - 1 - peel_level
            z_in = idx.z <= peel_level or idx.z >= dim.z - 1 - peel_level
            return x_in or y_in or z_in
        else:
            x_in = idx.x >= peel_level and idx.x <= dim.x - 1 - peel_level
            y_in = idx.y >= peel_level and idx.y <= dim.y - 1 - peel_level
            z_in = idx.z >= peel_level and idx.z <= dim.z - 1 - peel_level
            return x_in and y_in and z_in

    def create_level_mask(level, width):
        """Create a mask for a specific refinement level."""
        refinement = 2**level
        level_dim = neon.Index_3d(grid_shape[0] // refinement, grid_shape[1] // refinement, grid_shape[2] // refinement)

        if level == 0:
            level_dim = neon.Index_3d(*grid_shape)

        mask = np.zeros((level_dim.x, level_dim.y, level_dim.z), dtype=np.int32)

        for i in range(level_dim.x):
            for j in range(level_dim.y):
                for k in range(level_dim.z):
                    idx = neon.Index_3d(i, j, k)
                    if should_include_point(level_dim, idx, width, True):
                        mask[i, j, k] = 1

        return np.ascontiguousarray(mask, dtype=np.int32)

    def create_all_levels():
        """Create all refinement levels."""
        levels = []

        for i in range(num_levels - 1):
            level_mask = create_level_mask(i, peel_width)
            levels.append(level_mask)

        coarsest_level = num_levels - 1
        refinement = 2**coarsest_level
        coarsest_dim = neon.Index_3d(grid_shape[0] // refinement + 1, grid_shape[1] // refinement + 1, grid_shape[2] // refinement + 1)
        coarsest_mask = np.ones((coarsest_dim.x, coarsest_dim.y, coarsest_dim.z), dtype=np.int32)
        levels.append(np.ascontiguousarray(coarsest_mask, dtype=np.int32))

        return levels

    levels = create_all_levels()
    origins = [neon.Index_3d(0, 0, 0)] * len(levels)

    grid = multires_grid_factory(
        grid_shape,
        velocity_set=velocity_set,
        sparsity_pattern_list=levels,
        sparsity_pattern_origins=origins,
    )

    return grid, levels


def create_lid_driven_cavity_grid(grid_shape, velocity_set):
    """
    Create a simple 2-level grid for lid-driven cavity simulation.
    Fine resolution in the center, coarse at the boundaries.
    """
    num_levels = 2

    coarse_level = np.ones((grid_shape[0] // 2, grid_shape[1] // 2, grid_shape[2] // 2), dtype=np.int32)

    fine_size = min(40, grid_shape[0] // 3)
    fine_level = np.ones((fine_size, fine_size, fine_size), dtype=np.int32)
    fine_level = np.ascontiguousarray(fine_level, dtype=np.int32)

    levels = [fine_level, coarse_level]

    fine_origin_offset = (grid_shape[0] - fine_size) // 2
    level_origins = [(fine_origin_offset, fine_origin_offset, fine_origin_offset), (0, 0, 0)]

    grid = multires_grid_factory(
        grid_shape,
        velocity_set=velocity_set,
        sparsity_pattern_list=levels,
        sparsity_pattern_origins=[neon.Index_3d(*origin) for origin in level_origins],
    )

    return grid, levels


def setup_boundary_conditions_nested(grid, velocity_set):
    """Setup boundary conditions for nested refinement problem."""
    num_levels = grid.count_levels

    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)

    lid_indices = box_no_edge["top"]
    wall_indices = [box["bottom"][i] + box["left"][i] + box["right"][i] + box["front"][i] + box["back"][i] for i in range(velocity_set.d)]
    wall_indices = np.unique(np.array(wall_indices), axis=-1).tolist()

    lid_per_level = [lid_indices] + [[] for _ in range(num_levels - 1)]
    walls_per_level = [wall_indices] + [[] for _ in range(num_levels - 1)]

    return lid_per_level, walls_per_level


def setup_boundary_conditions_lid_driven(grid, velocity_set):
    """Setup boundary conditions for lid-driven cavity problem."""
    coarsest_level = grid.count_levels - 1

    box = grid.bounding_box_indices(shape=grid.level_to_shape(coarsest_level))
    box_no_edge = grid.bounding_box_indices(shape=grid.level_to_shape(coarsest_level), remove_edges=True)

    lid_indices = box_no_edge["top"]
    wall_indices = [box["bottom"][i] + box["left"][i] + box["right"][i] + box["front"][i] + box["back"][i] for i in range(velocity_set.d)]
    wall_indices = np.unique(np.array(wall_indices), axis=-1).tolist()

    lid_per_level = [[], lid_indices]
    walls_per_level = [[], wall_indices]

    return lid_per_level, walls_per_level


def print_grid_info(grid, levels, args, velocity_set):
    """Print detailed information about the multires grid."""
    print("\nGrid Information:")
    print(f"  Number of levels: {grid.count_levels}")
    print(f"  Refinement factor: {grid.refinement_factor}")

    total_active_cells = 0
    for i, level in enumerate(levels):
        active_cells = np.sum(level)
        refinement = 2**i if args.levels > 2 else (1 if i == 0 else 2)
        level_shape = grid.level_to_shape(i) if hasattr(grid, "level_to_shape") else level.shape
        print(f"  Level {i}: {level_shape} ({active_cells:,} active cells, refinement 1/{refinement})")
        total_active_cells += active_cells

    print(f"  Total active cells: {total_active_cells:,}")
    return total_active_cells


def run_simulation(args, velocity_set):
    """Run the multires simulation and return performance statistics."""
    grid_shape = (args.cube_edge, args.cube_edge, args.cube_edge)

    print(f"\nSetting up {args.levels}-level multires grid...")

    grid, levels = create_multires_grid(grid_shape, velocity_set, args.levels)

    if args.levels == 2:
        lid_indices, wall_indices = setup_boundary_conditions_lid_driven(grid, velocity_set)
    else:
        lid_indices, wall_indices = setup_boundary_conditions_nested(grid, velocity_set)

    total_active_cells = print_grid_info(grid, levels, args, velocity_set)

    boundary_conditions = [
        EquilibriumBC(rho=1.0, u=(args.velocity, 0.0, 0.0), indices=lid_indices),
        FullwayBounceBackBC(indices=wall_indices),
    ]

    characteristic_length = grid_shape[0] - 1
    viscosity = args.velocity * characteristic_length / args.reynolds
    omega = 1.0 / (3.0 * viscosity + 0.5)

    print("\nPhysical Parameters:")
    print(f"  Characteristic length: {characteristic_length}")
    print(f"  Kinematic viscosity: {viscosity:.6f}")
    print(f"  Relaxation parameter (ω): {omega:.6f}")

    print("\nInitializing simulation manager...")
    sim = xlb.helper.MultiresSimulationManager(
        omega=omega,
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type=args.collision_type,
    )

    print("\nStarting simulation...")
    print("Progress: ", end="", flush=True)

    wp.synchronize()
    start_time = time.time()

    progress_interval = max(1, args.num_steps // 20)

    for step in range(args.num_steps):
        sim.step()

        if step % progress_interval == 0 or step == args.num_steps - 1:
            progress = (step + 1) / args.num_steps * 100
            print(f"{progress:.0f}%", end=" ", flush=True)

    wp.synchronize()
    elapsed_time = time.time() - start_time

    print(f"\nSimulation completed in {elapsed_time:.4f} seconds")

    return {
        "time": elapsed_time,
        "num_levels": grid.count_levels,
        "grid_shape": grid_shape,
        "levels": args.levels,
        "total_active_cells": total_active_cells,
        "omega": omega,
        "viscosity": viscosity,
    }


def calculate_performance_metrics(args, stats):
    """Calculate MLUPS and other performance metrics."""
    cube_edge = args.cube_edge
    num_steps = args.num_steps
    elapsed_time = stats["time"]
    num_levels = stats["num_levels"]
    total_active_cells = stats["total_active_cells"]

    effective_fine_steps = num_steps * (2 ** (num_levels - 1))
    total_lattice_updates = total_active_cells * effective_fine_steps
    mlups = (total_lattice_updates / elapsed_time) / 1e6

    # Calculate efficiency metrics (compare to equivalent single-level simulation)
    equivalent_single_level_cells = cube_edge**3
    single_level_updates = equivalent_single_level_cells * num_steps
    single_level_mlups = (single_level_updates / elapsed_time) / 1e6
    efficiency = single_level_mlups / mlups if mlups > 0 else 0

    return {
        "EMLUPS": mlups,
        "single_level_MLUPS": single_level_mlups,
        "efficiency": efficiency,
        "effective_fine_steps": effective_fine_steps,
        "total_lattice_updates": total_lattice_updates,
        "lattice_updates_per_second": total_lattice_updates / elapsed_time,
        "time_per_step": elapsed_time / num_steps,
    }


def print_results(args, stats, performance):
    """Print comprehensive simulation results."""
    print("\n" + "=" * 80)
    print("PERFORMANCE RESULTS")
    print("=" * 80)

    print("Simulation Summary:")
    print(f"  Cube edge: {args.cube_edge} ({stats['total_active_cells']:,} active cells)")
    print(f"  Refinement: {stats['levels']} levels ({'simple' if stats['levels'] == 2 else 'nested'})")
    print(f"  Levels: {stats['num_levels']}")
    print(f"  Collision: {args.collision_type}")
    print(f"  Steps: {args.num_steps:,} coarse → {performance['effective_fine_steps']:,} effective fine")
    print(f"  Total updates: {performance['total_lattice_updates']:,}")

    print("\nTiming Results:")
    print(f"  Total time: {stats['time']:.4f} seconds")
    print(f"  Time per step: {performance['time_per_step']:.6f} seconds")
    print(f"  Updates/second: {performance['lattice_updates_per_second']:,.0f}")

    print("\nPerformance Metrics:")
    print(f"  Effective MLUPS: {performance['EMLUPS']:.2f}")
    print(f"  Single-level MLUPS: {performance['single_level_MLUPS']:.2f}")
    print(f"  Multires efficiency: {performance['efficiency']:.1%}")

    print("\nPhysical Parameters:")
    print(f"  Reynolds number: {args.reynolds:,.0f}")
    print(f"  Relaxation parameter: {stats['omega']:.6f}")
    print(f"  Kinematic viscosity: {stats['viscosity']:.6f}")

    print("=" * 80)


def show_examples():
    """Show copy-paste ready examples."""
    print("\nCopy-paste ready examples:")
    print("  # Basic BGK test (uses D3Q19) - 2 levels, simple")
    print("  python examples/performance/mlups_3d_multires.py 128 50 neon fp32/fp32")
    print()
    print("  # KBC test (uses D3Q27) - 2 levels, more accurate collision model")
    print("  python examples/performance/mlups_3d_multires.py 128 50 neon fp32/fp32 --collision_type KBC")
    print()
    print("  # Nested refinement - 4 levels, complex grid structure")
    print("  python examples/performance/mlups_3d_multires.py 128 50 neon fp32/fp32 --levels 4")
    print()
    print("  # Verbose output to see all details")
    print("  python examples/performance/mlups_3d_multires.py 128 50 neon fp32/fp32 --verbose")
    print()
    print("  # Larger grid for performance testing")
    print("  python examples/performance/mlups_3d_multires.py 256 100 neon fp32/fp32 --collision_type KBC --levels 3")
    print()


def main():
    """Main function to run the MLUPS benchmark."""
    import sys

    # Show examples if no arguments provided
    if len(sys.argv) == 1:
        print("MLUPS Benchmark for 3D Multires Lattice Boltzmann Method Simulation")
        print("=" * 70)
        show_examples()
        print("For full help: python examples/performance/mlups_3d_multires.py -h")
        return 0

    try:
        args = parse_arguments()

        velocity_set = setup_simulation(args)

        print_simulation_header(args, velocity_set)

        stats = run_simulation(args, velocity_set)
        performance = calculate_performance_metrics(args, stats)
        print_results(args, stats, performance)

        if args.verbose:
            print("\nDetailed Statistics:")
            print(f"  Args: {vars(args)}")
            print(f"  Stats: {stats}")
            print(f"  Performance: {performance}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose if "args" in locals() else False:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

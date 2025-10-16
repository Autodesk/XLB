# Lid Drive Cavity using out-of-core memory with XLB library

import os
import warp as wp
import numpy as np
from tqdm import tqdm
import logging
import mpi4py  # TODO: actually learn how mpi works...

mpi4py.rc.thread_level = "serialized"  # or 'funneled'
import mpi4py.MPI as MPI
import argparse
import math

wp.init()
wp.clear_kernel_cache()

# Import xlb
import xlb
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import EquilibriumBC, FullwayBounceBackBC
from xlb.operator.boundary_masker import IndicesBoundaryMasker
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import Macroscopic

# Local ooc imports
from ds import OOCGrid
from operators import UniformInitializer
from subroutine import (
    PrepareFieldsSubroutine,
    StepperSubroutine,
    RenderQCriterionSubroutine,
    VolumeSaverSubroutine,
)

# Make command line parser
parser = argparse.ArgumentParser(description="Lid driven cavity simulation")
parser.add_argument("--output_directory", type=str, default="ldc_output", help="Output directory")
parser.add_argument("--base_velocity", type=float, default=0.06, help="Base velocity")
parser.add_argument("--shape", type=str, default="(256, 256, 256)", help="Shape")
parser.add_argument("--tau", type=float, default=0.501, help="Tau")
parser.add_argument("--nr_steps", type=int, default=131072, help="Nr steps")
parser.add_argument("--save_q_criterion_frequency", type=int, default=128, help="Save q criterion frequency")
parser.add_argument("--q_criterion_threshold", type=float, default=1e-6, help="Q criterion threshold")
parser.add_argument("--save_volume_debug", type=bool, default=False, help="Save volume as vtk to debug")
parser.add_argument("--collision", type=str, default="SmagorinskyLESBGK", help="Collision")
parser.add_argument("--equilibrium", type=str, default="Quadratic", help="Equilibrium")
parser.add_argument("--velocity_set", type=str, default="D3Q19", help="Velocity set")
parser.add_argument("--ooc_block_shape", type=str, default="(128, 128, 128)", help="OOC block shape")
parser.add_argument("--ooc_ghost_cell_thickness", type=int, default=16, help="OOC ghost cell thickness")
parser.add_argument("--nr_streams", type=int, default=2, help="Nr streams")
parser.add_argument("--comm", type=bool, default=True, help="Comm")
args = parser.parse_args()

if __name__ == "__main__":
    # Set parameters
    output_directory = args.output_directory
    base_velocity = args.base_velocity
    shape = eval(args.shape)
    tau = args.tau
    nr_steps = args.nr_steps
    if args.save_q_criterion_frequency is None:
        save_q_criterion_frequency = -1
    else:
        save_q_criterion_frequency = (args.save_q_criterion_frequency // args.ooc_ghost_cell_thickness) * args.ooc_ghost_cell_thickness
    q_criterion_threshold = args.q_criterion_threshold
    collision = args.collision
    equilibrium = args.equilibrium
    velocity_set = args.velocity_set
    ooc_block_shape = eval(args.ooc_block_shape)
    ooc_ghost_cell_thickness = args.ooc_ghost_cell_thickness
    nr_streams = args.nr_streams
    if args.comm:
        comm = MPI.COMM_WORLD
    else:
        comm = None

    # Get fluid properties needed for the simulation
    omega = 1.0 / tau
    density = 1.0
    nr_steps = (nr_steps // ooc_ghost_cell_thickness) * ooc_ghost_cell_thickness  # Make sure steps is divisible by ghost cell thickness

    # Make output directory
    os.makedirs(output_directory, exist_ok=True)

    # Make logging
    logging.basicConfig(level=logging.INFO)

    # Log the parameters
    logging.info(f"Base velocity: {base_velocity}")
    logging.info(f"Shape: {shape}")
    logging.info(f"Tau: {tau}")
    logging.info(f"Omega: {omega}")
    logging.info(f"Nr steps: {nr_steps}")
    logging.info(f"Save q criterion frequency: {save_q_criterion_frequency}")
    logging.info(f"Collision: {collision}")
    logging.info(f"Equilibrium: {equilibrium}")
    logging.info(f"Velocity set: {velocity_set}")
    logging.info(f"OOC block shape: {ooc_block_shape}")
    logging.info(f"OOC ghost cell thickness: {ooc_ghost_cell_thickness}")
    logging.info(f"Nr streams: {nr_streams}")

    # Set the compute backend NOTE: hard coded for now
    compute_backend = xlb.ComputeBackend.WARP

    # Set the precision policy NOTE: hard coded for now
    precision_policy = xlb.PrecisionPolicy.FP32FP32

    # Set the velocity set
    if velocity_set == "D3Q27":
        velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
    elif velocity_set == "D3Q19":
        velocity_set = xlb.velocity_set.D3Q19(precision_policy=precision_policy, compute_backend=compute_backend)
    else:
        raise ValueError("Invalid velocity set")

    # Initialize XLB
    xlb.init(
        velocity_set=velocity_set,
        default_backend=compute_backend,
        default_precision_policy=precision_policy,
    )

    # Make grid for constructing stepper
    grid = xlb.grid.WarpGrid(shape=shape)

    # Make boundary conditions
    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    lid = box_no_edge["top"]
    walls = [box["bottom"][i] + box["left"][i] + box["right"][i] + box["front"][i] + box["back"][i] for i in range(velocity_set.d)]
    walls = np.unique(np.array(walls), axis=-1).tolist()
    bc_top = EquilibriumBC(
        rho=density,
        u=(0.0, base_velocity, 0.0),
        indices=lid,
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    # bc_walls = HalfwayBounceBackBC(
    bc_walls = FullwayBounceBackBC(
        indices=walls,
    )
    boundary_conditions = [bc_walls, bc_top]
    indices_boundary_masker = IndicesBoundaryMasker(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )

    # Make stepper
    stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type=collision,
    )

    # Make other operators
    macroscopic = Macroscopic(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    quadratic_equilibrium = QuadraticEquilibrium(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    uniform_initializer = UniformInitializer(
        initial_rho=density,
        initial_u=(0.0, 0.0, 0.0),
    )

    # Make subroutines
    prepare_fields_subroutine = PrepareFieldsSubroutine(
        initializer=uniform_initializer,
        equilibrium=quadratic_equilibrium,
        boundary_conditions=boundary_conditions,
        indices_boundary_masker=indices_boundary_masker,
        nr_streams=nr_streams,
    )
    stepper_subroutine = StepperSubroutine(
        stepper=stepper,
        omega=omega,
        nr_streams=nr_streams,
    )
    volume_saver_subroutine = VolumeSaverSubroutine(
        nr_streams=1,
    )
    render_q_criterion_subroutine = RenderQCriterionSubroutine(
        macroscopic=macroscopic,
        nr_streams=1,
    )

    # Make OOC
    ooc_grid = OOCGrid(
        shape=shape,
        block_shape=ooc_block_shape,
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0 / shape[0], 1.0 / shape[1], 1.0 / shape[2]),
        ghost_cell_thickness=ooc_ghost_cell_thickness,
        comm=comm,
    )

    # Initialize boxes for the OOC
    ooc_grid.initialize_boxes(
        name="f",
        dtype=wp.float32,
        cardinality=velocity_set.q,
        ordering="SOA",
    )
    ooc_grid.initialize_boxes(
        name="boundary_id",
        dtype=wp.uint8,
        cardinality=1,
        ordering="SOA",
    )
    ooc_grid.initialize_boxes(
        name="missing_mask",
        dtype=wp.bool,
        cardinality=velocity_set.q,
        ordering="SOA",
    )

    # Allocate ooc
    ooc_grid.allocate()

    # Make pixel buffer (UHD)
    pixel_buffer = wp.zeros((2160, 3840, 4), dtype=wp.float32)
    depth_buffer = wp.zeros((2160, 3840), dtype=wp.float32)

    # Prepare fields
    prepare_fields_subroutine(ooc_grid)

    # Save fields
    volume_saver_subroutine(
        ooc_grid,
        field_names=["f", "boundary_id", "missing_mask"],
        file_name=os.path.join(output_directory, "initial"),
    )

    # Start simulation
    logging.info("Starting simulation")
    for i in tqdm(range(nr_steps // ooc_ghost_cell_thickness)):
        # Perform stepper
        stepper_subroutine(ooc_grid)

        # Save volume and render q criterion
        if (i * ooc_ghost_cell_thickness) % save_q_criterion_frequency == 0 and save_q_criterion_frequency != -1:
            # Calculate camera position for orbit
            total_frames = nr_steps // save_q_criterion_frequency
            current_frame = i * ooc_ghost_cell_thickness // save_q_criterion_frequency
            angle = (current_frame / total_frames) * 2 * math.pi  # 0 to 2π

            # Camera parameters
            radius = 1.3  # Distance from center
            center = (0.5, 0.5, 0.5)  # Center of domain

            # Calculate camera position
            camera_x = center[0] + radius * math.cos(angle)
            camera_y = center[1] - radius * math.sin(angle)
            camera_z = center[2]

            pixel_buffer.fill_(0.0)
            depth_buffer.fill_(10.0)
            render_q_criterion_subroutine(
                ooc_grid,
                os.path.join(output_directory, f"q_criterion_{i:06d}"),  # Zero-padded frame numbers
                pixel_buffer,
                depth_buffer,
                camera_pos=(camera_x, camera_y, camera_z),
                camera_target=center,  # Look at center
                camera_up=(0.0, 0.0, 1.0),  # Keep camera upright
                fov_degrees=60.0,
                ambient_intensity=0.05,
                edge_sharpness=1.0,
                gamma=1.0,
                q_criterion_threshold=q_criterion_threshold,
                vmin=0.0,
                vmax=0.01,
            )

            # Save
            if args.save_volume_debug:
                volume_saver_subroutine(
                    ooc_grid, field_names=["f", "boundary_id", "missing_mask"], file_name=os.path.join(output_directory, f"time_{str(i).zfill(5)}")
                )

# Wind tunnel simulation using the XLB library

import os
import numpy as np
import warp as wp
from tqdm import tqdm
import logging
import mpi4py  # TODO: actually learn how mpi works...

mpi4py.rc.thread_level = "serialized"  # or 'funneled'
import mpi4py.MPI as MPI
import argparse

wp.init()
wp.clear_kernel_cache()

# Import xlb
import xlb
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import FullwayBounceBackBC
from xlb.operator.boundary_masker import IndicesBoundaryMasker
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import Macroscopic

# Local ooc imports
from ds import OOCGrid
from operators import (
    ClampField,
    UniformInitializer,
    InitializeTargetDensity,
    L2Loss,
)
from subroutine import (
    PrepareFieldsSubroutine,
    VolumeSaverSubroutine,
    ForwardStepperSubroutine,
    BackwardStepperSubroutine,
    ForwardRhoLossSubroutine,
    BackwardRhoLossSubroutine,
    GradientDescentSubroutine,
    InitializeFieldSubroutine,
)

# Make command line parser
parser = argparse.ArgumentParser(description="Differential LBM")
parser.add_argument("--output_directory", type=str, default="autodiff_output", help="Output directory")
parser.add_argument("--final_stl_file", type=str, default="assets/nvidia_new.stl", help="STL file to match at the end of the simulation")
parser.add_argument("--max_base_velocity", type=float, default=0.02, help="Base velocity")
parser.add_argument("--shape", type=str, default="(128, 128, 128)", help="Shape")
parser.add_argument("--tau", type=float, default=0.53, help="Tau")
parser.add_argument("--nr_optimization_steps", type=int, default=1024, help="Nr optimization steps")
parser.add_argument("--nr_steps", type=int, default=256, help="Nr steps")
parser.add_argument("--checkpoint_frequency", type=int, default=16, help="Checkpoint frequency")
parser.add_argument("--save_state_frequency", type=int, default=16, help="Save volume frequency")
parser.add_argument("--collision", type=str, default="BGK", help="Collision")
parser.add_argument("--equilibrium", type=str, default="Quadratic", help="Equilibrium")
parser.add_argument("--velocity_set", type=str, default="D3Q19", help="Velocity set")
parser.add_argument("--ooc_block_shape", type=str, default="(64, 64, 64)", help="OOC block shape")
parser.add_argument("--nr_streams", type=int, default=1, help="Nr streams")
parser.add_argument("--comm", type=bool, default=True, help="Comm")
args = parser.parse_args()


def forward(
    ooc_grid,
    loss,
    checkpoint_frequency,
    nr_checkpoints,
    forward_stepper_subroutine,
    forward_rho_loss_subroutine,
):
    # Zero the loss
    loss.zero_()

    # Perform forward pass
    for i in range(nr_checkpoints):
        # Perform forward step
        forward_stepper_subroutine(
            ooc_grid,
            nr_steps=checkpoint_frequency,
            f_input_name=f"f_{str(i).zfill(4)}",
            f_output_name=f"f_{str(i + 1).zfill(4)}",
            boundary_id_name="boundary_id",
            missing_mask_name="missing_mask",
        )

    # Compute loss
    forward_rho_loss_subroutine(
        ooc_grid,
        f_name=f"f_{str(nr_checkpoints).zfill(4)}",
        target_rho_name="target_density",
        loss=loss,
    )


def backward(
    ooc_grid,
    loss,
    checkpoint_frequency,
    nr_checkpoints,
    backward_stepper_subroutine,
    backward_rho_loss_subroutine,
):
    # Set the loss gradient
    loss.grad.fill_(1.0)

    # Perform backward pass
    backward_rho_loss_subroutine(
        ooc_grid,
        f_name=f"f_{str(nr_checkpoints).zfill(4)}",
        adj_f_name="adj_f",
        target_rho_name="target_density",
        loss=loss,
    )

    # Perform backward step
    for i in range(nr_checkpoints - 1, -1, -1):
        # Perform backward step
        backward_stepper_subroutine(
            ooc_grid,
            nr_steps=checkpoint_frequency,
            f_input_name=f"f_{str(i).zfill(4)}",
            adj_f_name="adj_f",
            boundary_id_name="boundary_id",
            missing_mask_name="missing_mask",
        )


if __name__ == "__main__":
    # Set parameters
    output_directory = args.output_directory
    final_stl_file = args.final_stl_file
    max_base_velocity = args.max_base_velocity
    shape = eval(args.shape)
    tau = args.tau
    nr_optimization_steps = args.nr_optimization_steps
    nr_steps = args.nr_steps
    checkpoint_frequency = args.checkpoint_frequency
    if args.save_state_frequency is None:
        save_state_frequency = -1
    else:
        save_state_frequency = args.save_state_frequency
    collision = args.collision
    equilibrium = args.equilibrium
    velocity_set = args.velocity_set
    ooc_block_shape = eval(args.ooc_block_shape)
    ooc_ghost_cell_thickness = args.checkpoint_frequency * 2
    nr_streams = args.nr_streams
    if args.comm:
        comm = MPI.COMM_WORLD
    else:
        comm = None

    # Get fluid properties needed for the simulation
    omega = 1.0 / tau
    density = 1.0
    nr_steps = (nr_steps // checkpoint_frequency) * checkpoint_frequency
    nr_checkpoints = nr_steps // checkpoint_frequency

    # Make output directory
    os.makedirs(output_directory, exist_ok=True)

    # Make logging
    logging.basicConfig(level=logging.INFO)

    # Log the parameters
    logging.info(f"Output directory: {output_directory}")
    logging.info(f"Final STL file: {final_stl_file}")
    logging.info(f"Max base velocity: {max_base_velocity}")
    logging.info(f"Shape: {shape}")
    logging.info(f"Tau: {tau}")
    logging.info(f"Omega: {omega}")
    logging.info(f"Nr steps: {nr_steps}")
    logging.info(f"Nr optimization steps: {nr_optimization_steps}")
    logging.info(f"Save state frequency: {save_state_frequency}")
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
    walls = [box["top"][i] + box["bottom"][i] + box["left"][i] + box["right"][i] + box["front"][i] + box["back"][i] for i in range(velocity_set.d)]
    walls = np.unique(np.array(walls), axis=-1).tolist()
    bc_walls = FullwayBounceBackBC(
        indices=walls,
    )
    boundary_conditions = [bc_walls]
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
    initialize_target_density = InitializeTargetDensity(file_path=final_stl_file, background_density=density, mesh_density=density + 0.0025)
    l2_loss = L2Loss()

    # Make subroutines
    prepare_fields_subroutine = PrepareFieldsSubroutine(
        initializer=uniform_initializer,
        equilibrium=quadratic_equilibrium,
        boundary_conditions=boundary_conditions,
        indices_boundary_masker=indices_boundary_masker,
        nr_streams=nr_streams,
    )
    forward_stepper_subroutine = ForwardStepperSubroutine(
        stepper=stepper,
        omega=omega,
        nr_streams=nr_streams,
    )
    backward_stepper_subroutine = BackwardStepperSubroutine(
        stepper=stepper,
        omega=omega,
        nr_streams=nr_streams,
    )
    forward_rho_loss_subroutine = ForwardRhoLossSubroutine(
        macroscopic=macroscopic,
        loss=l2_loss,
        nr_streams=nr_streams,
    )
    backward_rho_loss_subroutine = BackwardRhoLossSubroutine(
        macroscopic=macroscopic,
        loss=l2_loss,
        nr_streams=nr_streams,
    )
    volume_saver_subroutine = VolumeSaverSubroutine(
        nr_streams=1,
    )
    initialize_target_density_subroutine = InitializeFieldSubroutine(
        initializer=initialize_target_density,
        nr_streams=nr_streams,
    )
    volume_saver_subroutine = VolumeSaverSubroutine()
    gradient_descent_subroutine = GradientDescentSubroutine(
        clamp_field=ClampField(),
        nr_streams=nr_streams,
    )

    # Make OOC grid
    ooc_grid = OOCGrid(
        shape=shape,
        block_shape=ooc_block_shape,
        origin=(-1.0, -1.0, -1.0),
        spacing=(2.0 / shape[0], 2.0 / shape[1], 2.0 / shape[2]),
        ghost_cell_thickness=ooc_ghost_cell_thickness,
        comm=comm,
    )

    # Make loss
    loss = wp.zeros((1,), dtype=float, requires_grad=True)

    # Make min and max values for clamping
    min_val = wp.from_numpy(
        0.9 * np.array(velocity_set.w),
        dtype=wp.float32,
    )
    max_val = wp.from_numpy(
        1.1 * np.array(velocity_set.w),
        dtype=wp.float32,
    )

    # Initialize boxes for the OOC
    ooc_grid.initialize_boxes(
        name="target_density",
        dtype=wp.float32,
        cardinality=1,
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
    ooc_grid.initialize_boxes(
        name="adj_f",
        dtype=wp.float32,
        cardinality=velocity_set.q,
        ordering="SOA",
    )
    for i in range(nr_checkpoints + 1):
        ooc_grid.initialize_boxes(
            name=f"f_{str(i).zfill(4)}",
            dtype=wp.float32,
            cardinality=velocity_set.q,
            ordering="SOA",
        )
    # Allocate ooc
    ooc_grid.allocate()
    print(f"nr gigs: {ooc_grid.nbytes // 1e9}")

    # Prepare fields
    prepare_fields_subroutine(
        ooc_grid,
        f_name="f_0000",
    )

    # Initialize target velocity norm
    initialize_target_density_subroutine(
        ooc_grid,
        field_name="target_density",
    )

    # Save target density
    volume_saver_subroutine(
        ooc_grid,
        field_names=["target_density", "f_0000", "boundary_id", "missing_mask"],
        file_name=os.path.join(output_directory, "target_density"),
    )

    # Start optimization
    logging.info("Starting optimization")
    for i in tqdm(range(nr_optimization_steps)):
        # Perform forward pass
        forward(
            ooc_grid,
            loss,
            checkpoint_frequency,
            nr_checkpoints,
            forward_stepper_subroutine,
            forward_rho_loss_subroutine,
        )

        # Perform backward pass
        backward(
            ooc_grid,
            loss,
            checkpoint_frequency,
            nr_checkpoints,
            backward_stepper_subroutine,
            backward_rho_loss_subroutine,
        )

        # Perform gradient descent
        gradient_descent_subroutine(
            ooc_grid,
            field_name="f_0000",
            adj_field_name="adj_f",
            learning_rate=0.001,
            min_val=min_val,
            max_val=max_val,
        )

        # Print loss
        logging.info(f"Loss: {loss.numpy()[0]}")

        # Check if loss is nan
        if np.isnan(loss.numpy()[0]):
            logging.info("Loss is nan")
            break

        # Save volume
        if i % save_state_frequency == 0 and i != 0:
            volume_saver_subroutine(
                ooc_grid,
                field_names=["adj_f"],
                file_name=os.path.join(output_directory, "adj_f"),
            )
            for j in range(nr_checkpoints + 1):
                volume_saver_subroutine(
                    ooc_grid,
                    field_names=[f"f_{str(j).zfill(4)}"],
                    file_name=os.path.join(output_directory, f"state_{str(j).zfill(4)}"),
                )

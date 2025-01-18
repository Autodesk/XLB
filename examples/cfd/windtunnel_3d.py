import xlb
import trimesh
import time
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    HalfwayBounceBackBC,
    FullwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
)
from xlb.operator.force.momentum_transfer import MomentumTransfer
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
import warp as wp
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# -------------------------- Helper Functions --------------------------


def plot_drag_coefficient(time_steps, drag_coefficients):
    """
    Plot the drag coefficient with various moving averages.

    Args:
        time_steps (list): List of time steps.
        drag_coefficients (list): List of drag coefficients.
    """
    # Convert lists to numpy arrays for processing
    time_steps_np = np.array(time_steps)
    drag_coefficients_np = np.array(drag_coefficients)

    # Define moving average windows
    windows = [10, 100, 1000, 10000, 100000]
    labels = ["MA 10", "MA 100", "MA 1,000", "MA 10,000", "MA 100,000"]

    plt.figure(figsize=(12, 8))
    plt.plot(time_steps_np, drag_coefficients_np, label="Raw", alpha=0.5)

    for window, label in zip(windows, labels):
        if len(drag_coefficients_np) >= window:
            ma = np.convolve(drag_coefficients_np, np.ones(window) / window, mode="valid")
            plt.plot(time_steps_np[window - 1 :], ma, label=label)

    plt.ylim(-1.0, 1.0)
    plt.legend()
    plt.xlabel("Time step")
    plt.ylabel("Drag coefficient")
    plt.title("Drag Coefficient Over Time with Moving Averages")
    plt.savefig("drag_coefficient_ma.png")
    plt.close()


def define_boundary_indices(grid, velocity_set, grid_shape):
    """
    Define inlet, outlet, walls, and load the mesh for the car.

    Args:
        grid: XLB grid object.
        velocity_set: Velocity set object.
        grid_shape (tuple): Shape of the grid.

    Returns:
        inlet (list): Inlet boundary indices.
        outlet (list): Outlet boundary indices.
        walls (list): Wall boundary indices.
        car_vertices (numpy.ndarray): Transformed mesh vertices.
        car_cross_section (float): Cross-sectional area of the car.
    """
    # Bounding box indices
    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    inlet = box_no_edge["left"]
    outlet = box_no_edge["right"]
    walls = [box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(velocity_set.d)]
    walls = np.unique(np.array(walls), axis=-1).tolist()

    # Load the mesh (replace with your own mesh)
    stl_filename = "../stl-files/DrivAer-Notchback.stl"
    mesh = trimesh.load_mesh(stl_filename, process=False)
    mesh_vertices = mesh.vertices

    # Transform the mesh points to align with the grid
    mesh_vertices -= mesh_vertices.min(axis=0)
    mesh_extents = mesh_vertices.max(axis=0)
    length_phys_unit = mesh_extents.max()
    length_lbm_unit = grid_shape[0] / 4
    dx = length_phys_unit / length_lbm_unit
    mesh_vertices = mesh_vertices / dx
    shift = np.array([grid_shape[0] / 4, (grid_shape[1] - mesh_extents[1] / dx) / 2, 0.0])
    car_vertices = mesh_vertices + shift
    car_cross_section = np.prod(mesh_extents[1:]) / dx**2

    return inlet, outlet, walls, car_vertices, car_cross_section


def setup_boundary_conditions(inlet, outlet, walls, car_vertices, wind_speed):
    """
    Setup boundary conditions for the simulation.

    Args:
        inlet (list): Inlet boundary indices.
        outlet (list): Outlet boundary indices.
        walls (list): Wall boundary indices.
        car_vertices (numpy.ndarray): Transformed mesh vertices.
        wind_speed (float): Prescribed wind speed.

    Returns:
        boundary_conditions (list): List of boundary condition objects.
    """
    bc_left = RegularizedBC("velocity", prescribed_value=(wind_speed, 0.0, 0.0), indices=inlet)
    bc_walls = FullwayBounceBackBC(indices=walls)
    bc_do_nothing = ExtrapolationOutflowBC(indices=outlet)
    bc_car = HalfwayBounceBackBC(mesh_vertices=car_vertices)
    boundary_conditions = [bc_walls, bc_left, bc_do_nothing, bc_car]
    return boundary_conditions


def initialize_fields(stepper, grid, velocity_set, precision_policy, compute_backend):
    """
    Initialize the distribution functions with a random velocity field.

    Args:
        stepper: IncompressibleNavierStokesStepper object.
        grid: XLB grid object.
        velocity_set: Velocity set object.
        precision_policy: PrecisionPolicy object.
        compute_backend: ComputeBackend enum.

    Returns:
        f_0, f_1, bc_mask, missing_mask: Initialized fields and masks.
    """
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

    # Initialize with random velocity field
    shape = (velocity_set.d,) + grid.shape
    np.random.seed(0)
    u_init = np.random.random(shape)
    if compute_backend == ComputeBackend.JAX:
        u_init = 1e-2 * u_init
        u_init = jnp.array(u_init, dtype=jnp.float32)
    else:
        u_init = wp.array(1e-2 * u_init, dtype=precision_policy.compute_precision.wp_dtype)

    f_0 = xlb.helper.initialize_eq(f_0, grid, velocity_set, precision_policy, compute_backend, u=u_init)

    return f_0, f_1, bc_mask, missing_mask


def post_process(
    step,
    f_0,
    f_1,
    grid_shape,
    macro,
    momentum_transfer,
    missing_mask,
    bc_mask,
    wind_speed,
    car_cross_section,
    drag_coefficients,
    lift_coefficients,
    time_steps,
):
    """
    Post-process simulation data: save fields, compute forces, and plot drag coefficient.

    Args:
        step (int): Current time step.
        f_current: Current distribution function.
        grid_shape (tuple): Shape of the grid.
        macro: Macroscopic operator object.
        momentum_transfer: MomentumTransfer operator object.
        missing_mask: Missing mask from stepper.
        bc_mask: Boundary condition mask from stepper.
        wind_speed (float): Prescribed wind speed.
        car_cross_section (float): Cross-sectional area of the car.
        drag_coefficients (list): List to store drag coefficients.
        lift_coefficients (list): List to store lift coefficients.
        time_steps (list): List to store time steps.
    """
    # Convert to JAX array if necessary
    if not isinstance(f_0, jnp.ndarray):
        f_0_jax = wp.to_jax(f_0)
    else:
        f_0_jax = f_0

    # Compute macroscopic quantities
    rho, u = macro(f_0_jax)

    # Remove boundary cells
    u = u[:, 1:-1, 1:-1, 1:-1]
    u_magnitude = jnp.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)

    fields = {"u_magnitude": u_magnitude}

    # Save fields in VTK format
    save_fields_vtk(fields, timestep=step)

    # Save the u_magnitude slice at the mid y-plane
    mid_y = grid_shape[1] // 2
    save_image(fields["u_magnitude"][:, mid_y, :], timestep=step)

    # Compute lift and drag
    boundary_force = momentum_transfer(f_0, f_1, bc_mask, missing_mask)
    drag = np.sqrt(boundary_force[0] ** 2 + boundary_force[1] ** 2)  # xy-plane
    lift = boundary_force[2]
    c_d = 2.0 * drag / (wind_speed**2 * car_cross_section)
    c_l = 2.0 * lift / (wind_speed**2 * car_cross_section)
    drag_coefficients.append(c_d)
    lift_coefficients.append(c_l)
    time_steps.append(step)

    # Plot drag coefficient
    plot_drag_coefficient(time_steps, drag_coefficients)


# -------------------------- Simulation Setup --------------------------

# Grid parameters
grid_size_x, grid_size_y, grid_size_z = 512, 128, 128
grid_shape = (grid_size_x, grid_size_y, grid_size_z)

# Simulation Configuration
compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32

velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
wind_speed = 0.02
num_steps = 100000
print_interval = 1000
post_process_interval = 1000

# Physical Parameters
Re = 50000.0
clength = grid_size_x - 1
visc = wind_speed * clength / Re
omega = 1.0 / (3.0 * visc + 0.5)

# Print simulation info
print("\n" + "=" * 50 + "\n")
print("Simulation Configuration:")
print(f"Grid size: {grid_size_x} x {grid_size_y} x {grid_size_z}")
print(f"Backend: {compute_backend}")
print(f"Velocity set: {velocity_set}")
print(f"Precision policy: {precision_policy}")
print(f"Prescribed velocity: {wind_speed}")
print(f"Reynolds number: {Re}")
print(f"Max iterations: {num_steps}")
print("\n" + "=" * 50 + "\n")

# Initialize XLB
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Create Grid
grid = grid_factory(grid_shape, compute_backend=compute_backend)

# Define Boundary Indices and Load Mesh
inlet, outlet, walls, car_vertices, car_cross_section = define_boundary_indices(grid, velocity_set, grid_shape)

# Setup Boundary Conditions
boundary_conditions = setup_boundary_conditions(inlet, outlet, walls, car_vertices, wind_speed)

# Setup Stepper
stepper = IncompressibleNavierStokesStepper(
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="KBC",
)

# Prepare Fields
f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

# Setup Momentum Transfer for Force Calculation
bc_car = boundary_conditions[-1]
momentum_transfer = MomentumTransfer(bc_car, compute_backend=compute_backend)

# Define Macroscopic Calculation
macro = Macroscopic(
    compute_backend=ComputeBackend.JAX,
    precision_policy=precision_policy,
    velocity_set=xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=ComputeBackend.JAX),
)

# Initialize Lists to Store Coefficients and Time Steps
time_steps = []
drag_coefficients = []
lift_coefficients = []

# -------------------------- Simulation Loop --------------------------

start_time = time.time()
for step in range(num_steps):
    # Perform simulation step
    f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
    f_0, f_1 = f_1, f_0  # Swap the buffers

    # Print progress at intervals
    if (step + 1) % print_interval == 0:
        elapsed_time = time.time() - start_time
        print(f"Iteration: {step + 1}/{num_steps} | Time elapsed: {elapsed_time:.2f}s")
        start_time = time.time()

    # Post-process at intervals and final step
    if (step % post_process_interval == 0) or (step == num_steps - 1):
        post_process(
            step,
            f_0,
            f_1,
            grid_shape,
            macro,
            momentum_transfer,
            missing_mask,
            bc_mask,
            wind_speed,
            car_cross_section,
            drag_coefficients,
            lift_coefficients,
            time_steps,
        )

print("Simulation completed successfully.")

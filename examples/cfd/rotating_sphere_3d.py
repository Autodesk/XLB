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
    DoNothingBC,
    HybridBC,
)
from xlb.operator.force.momentum_transfer import MomentumTransfer
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
import warp as wp
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator import Operator
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet


# -------------------------- Simulation Setup --------------------------

# Grid parameters
wp.clear_kernel_cache()
diam = 32
grid_size_x, grid_size_y, grid_size_z = 10 * diam, 7 * diam, 7 * diam
grid_shape = (grid_size_x, grid_size_y, grid_size_z)

# Simulation Configuration
compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32

velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
wind_speed = 0.04
num_steps = 100000
print_interval = 1000
post_process_interval = 1000

# Physical Parameters
Re = 200.0
visc = wind_speed * diam / Re
omega = 1.0 / (3.0 * visc + 0.5)

# Rotational speed parameters (see [1] which discusses the problem in terms of 2 non-dimensional parameters: Re and Omega)
# [1] J. Fluid Mech. (2016), vol. 807, pp. 62–86. c© Cambridge University Press 2016 doi:10.1017/jfm.2016.596
# \Omega = \omega * D / (2 U_\infty) where Omega is non-dimensional and omega is dimensional.
rot_rate_nondim = -0.2
rot_rate = 2.0 * wind_speed * rot_rate_nondim / diam

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

# Bounding box indices
box = grid.bounding_box_indices()
box_no_edge = grid.bounding_box_indices(remove_edges=True)
inlet = box_no_edge["left"]
outlet = box["right"]
walls = [box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(velocity_set.d)]
walls = np.unique(np.array(walls), axis=-1).tolist()

# Load the mesh (replace with your own mesh)
stl_filename = "examples/cfd/stl-files/sphere.stl"
mesh = trimesh.load_mesh(stl_filename, process=False)
mesh_vertices = mesh.vertices

# Transform the mesh points to be located in the right position in the wind tunnel
mesh_vertices -= mesh_vertices.min(axis=0)
mesh_extents = mesh_vertices.max(axis=0)
length_phys_unit = mesh_extents.max()
length_lbm_unit = grid_shape[1] / 7
dx = length_phys_unit / length_lbm_unit
mesh_vertices = mesh_vertices / dx
shift = np.array([grid_shape[0] / 3, (grid_shape[1] - mesh_extents[1] / dx) / 2, (grid_shape[2] - mesh_extents[2] / dx) / 2])
sphere = mesh_vertices + shift
diam = np.max(sphere.max(axis=0) - sphere.min(axis=0))
sphere_cross_section = np.pi * diam**2 / 4.0


# Define rotating boundary profile
def bc_profile():
    _u_vec = wp.vec(velocity_set.d, dtype=precision_policy.compute_precision.wp_dtype)
    angular_velocity = _u_vec(0.0, rot_rate, 0.0)
    origin_np = shift + diam / 2
    origin_wp = _u_vec(origin_np[0], origin_np[1], origin_np[2])

    @wp.func
    def bc_profile_warp(index: wp.vec3i, time: Any):
        x = wp.float32(index[0])
        y = wp.float32(index[1])
        z = wp.float32(index[2])
        surface_coord = _u_vec(x, y, z) - origin_wp
        return wp.cross(angular_velocity, surface_coord)

    return bc_profile_warp


# Define boundary conditions
bc_left = RegularizedBC("velocity", prescribed_value=(wind_speed, 0.0, 0.0), indices=inlet)
bc_do_nothing = DoNothingBC(indices=outlet)
# bc_sphere = HalfwayBounceBackBC(mesh_vertices=sphere, voxelization_method="ray", profile=bc_profile())
bc_sphere = HybridBC(
    bc_method="nonequilibrium_regularized", mesh_vertices=sphere, use_mesh_distance=True, voxelization_method="ray", profile=bc_profile()
)
# Not assining BC for walls makes them periodic.
boundary_conditions = [bc_left, bc_do_nothing, bc_sphere]


# Setup Stepper
stepper = IncompressibleNavierStokesStepper(
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="KBC",
)


# Defining an initializer for outlet only
class OutletInitializer(Operator):
    def __init__(
        self,
        wind_speed=None,
        grid_shape=None,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        self.wind_speed = wind_speed
        self.rho = 1.0
        self.grid_shape = grid_shape
        self.equilibrium = QuadraticEquilibrium(velocity_set=velocity_set, precision_policy=precision_policy, compute_backend=compute_backend)
        super().__init__(velocity_set, precision_policy, compute_backend)

    def _construct_warp(self):
        nx, ny, nz = self.grid_shape
        _q = self.velocity_set.q
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _rho = self.compute_dtype(self.rho)
        _u = _u_vec(self.wind_speed, 0.0, 0.0)
        _w = self.velocity_set.w

        # Construct the warp kernel
        @wp.kernel
        def kernel(f: wp.array4d(dtype=Any)):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # Set the velocity at the outlet (i.e. where i = nx-1)
            if index[0] == nx - 1:
                _feq = self.equilibrium.warp_functional(_rho, _u)
                for l in range(_q):
                    f[l, index[0], index[1], index[2]] = _feq[l]
            else:
                # In the rest of the domain, we assume zero velocity and equilibrium distribution.
                for l in range(_q):
                    f[l, index[0], index[1], index[2]] = _w[l]

        return None, kernel

    @Operator.register_backend(xlb.ComputeBackend.WARP)
    def warp_implementation(self, f):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f,
            ],
            dim=f.shape[1:],
        )
        return f


# Make initializer operator
initializer = OutletInitializer(
    wind_speed=wind_speed,
    grid_shape=grid_shape,
    velocity_set=velocity_set,
    precision_policy=precision_policy,
    compute_backend=compute_backend,
)

# Prepare Fields
f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields(initializer=initializer)


# -------------------------- Helper Functions --------------------------


def plot_coefficient(time_steps, coefficients, prefix="drag"):
    """
    Plot the drag coefficient with various moving averages.

    Args:
        time_steps (list): List of time steps.
        coefficients (list): List of force coefficients.
    """
    # Convert lists to numpy arrays for processing
    time_steps_np = np.array(time_steps)
    coefficients_np = np.array(coefficients)

    # Define moving average windows
    windows = [10, 100, 1000, 10000, 100000]
    labels = ["MA 10", "MA 100", "MA 1,000", "MA 10,000", "MA 100,000"]

    plt.figure(figsize=(12, 8))
    plt.plot(time_steps_np, coefficients_np, label="Raw", alpha=0.5)

    for window, label in zip(windows, labels):
        if len(coefficients_np) >= window:
            ma = np.convolve(coefficients_np, np.ones(window) / window, mode="valid")
            plt.plot(time_steps_np[window - 1 :], ma, label=label)

    plt.ylim(-1.0, 1.0)
    plt.legend()
    plt.xlabel("Time step")
    plt.ylabel("Drag coefficient")
    plt.title("Drag Coefficient Over Time with Moving Averages")
    plt.savefig(prefix + "_ma.png")
    plt.close()


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
    wp.synchronize()
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

    fields = {"ux": u[0], "uy": u[1], "uz": u[2], "u_magnitude": u_magnitude}

    # Save fields in VTK format
    # save_fields_vtk(fields, timestep=step)

    # Save the u_magnitude slice at the mid y-plane
    mid_y = grid_shape[1] // 2
    save_image(fields["u_magnitude"][:, mid_y, :], timestep=step)

    # Compute lift and drag
    boundary_force = momentum_transfer(f_0, f_1, bc_mask, missing_mask)
    drag = boundary_force[0]  # x-direction
    lift = boundary_force[2]
    cd = 2.0 * drag / (wind_speed**2 * car_cross_section)
    cl = 2.0 * lift / (wind_speed**2 * car_cross_section)
    print(f"CD={cd}, CL={cl}")
    drag_coefficients.append(cd)
    lift_coefficients.append(cl)
    time_steps.append(step)

    # Plot drag coefficient
    plot_coefficient(time_steps, drag_coefficients, prefix="drag")
    plot_coefficient(time_steps, lift_coefficients, prefix="lift")


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
    if step % print_interval == 0:
        elapsed_time = time.time() - start_time
        print(f"Iteration: {step}/{num_steps} | Time elapsed: {elapsed_time:.2f}s")
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
            sphere_cross_section,
            drag_coefficients,
            lift_coefficients,
            time_steps,
        )

print("Simulation completed successfully.")

import xlb
import time
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import RegularizedBC
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
from xlb.helper import initialize_eq
import warp as wp
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import json


# -------------------------- Helper Functions --------------------------


def vonKarman_loglaw_wall(yplus):
    vonKarmanConst = 0.41
    cplus = 5.5
    uplus = np.log(yplus) / vonKarmanConst + cplus
    return uplus


def get_dns_data():
    """
    Reference: DNS of Turbulent Channel Flow up to Re_tau=590, 1999,
    Physics of Fluids, vol 11, 943-945.
    https://turbulence.oden.utexas.edu/data/MKM/chan180/profiles/chan180.means
    """
    file_name = "examples/cfd/data/turbulent_channel_dns_data.json"
    with open(file_name, "r") as file:
        return json.load(file)


# -------------------------- Simulation Setup --------------------------

# Channel Parameter
channel_half_width = 50

# Define channel geometry based on h
grid_size_x = 6 * channel_half_width
grid_size_y = 3 * channel_half_width
grid_size_z = 2 * channel_half_width

# Grid parameters
grid_shape = (grid_size_x, grid_size_y, grid_size_z)

# Define flow regime
Re_tau = 180
u_tau = 0.001

# Compute viscosity and relaxation parameter omega
visc = u_tau * channel_half_width / Re_tau
omega = 1.0 / (3.0 * visc + 0.5)

# Runtime & compute_backend configurations
compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP64FP64
velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)
num_steps = 10000000
print_interval = 100000
post_process_interval = 100000

# Print simulation info
print("\n" + "=" * 50 + "\n")
print("Simulation Configuration:")
print(f"Grid size: {grid_size_x} x {grid_size_y} x {grid_size_z}")
print(f"Backend: {compute_backend}")
print(f"Velocity set: {velocity_set}")
print(f"Precision policy: {precision_policy}")
print(f"Reynolds number: {Re_tau}")
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


# Define Force Vector
def get_force(Re_tau, visc, channel_half_width, velocity_set):
    shape = (velocity_set.d,)
    force = np.zeros(shape)
    force[0] = Re_tau**2 * visc**2 / channel_half_width**3
    return force


force_vector = get_force(Re_tau, visc, channel_half_width, velocity_set)


# Define Boundary Indices
box = grid.bounding_box_indices(remove_edges=True)
walls = [box["bottom"][i] + box["top"][i] for i in range(velocity_set.d)]


# Define Boundary Conditions
def setup_boundary_conditions(walls, velocity_set, precision_policy):
    # No-slip boundary condition: velocity = (0, 0, 0)
    bc_walls = RegularizedBC("velocity", prescribed_value=(0.0, 0.0, 0.0), indices=walls)
    boundary_conditions = [bc_walls]
    return boundary_conditions


boundary_conditions = setup_boundary_conditions(walls, velocity_set, precision_policy)

# Setup Stepper
stepper = IncompressibleNavierStokesStepper(
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="KBC",
    force_vector=force_vector,
)

# Prepare Fields
f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()


# Initialize Fields with Random Velocity
shape = (velocity_set.d,) + grid.shape
np.random.seed(0)
u_init = np.random.random(shape)
if compute_backend == ComputeBackend.JAX:
    u_init = jnp.full(shape=shape, fill_value=1e-2 * u_init)
else:
    u_init = wp.array(1e-2 * u_init, dtype=precision_policy.compute_precision.wp_dtype)

f_0 = initialize_eq(f_0, grid, velocity_set, precision_policy, compute_backend, u=u_init)

# Define Macroscopic Calculation
macro = Macroscopic(
    compute_backend=ComputeBackend.JAX,
    precision_policy=precision_policy,
    velocity_set=xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=ComputeBackend.JAX),
)


# Post-Processing Function
def post_process(step, f_current, grid_shape, macro):
    # Convert to JAX array if necessary
    if not isinstance(f_current, jnp.ndarray):
        f_current = wp.to_jax(f_current)

    rho, u = macro(f_current)

    # Compute velocity magnitude
    u_magnitude = jnp.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    fields = {
        "rho": rho[0],
        "u_x": u[0],
        "u_y": u[1],
        "u_z": u[2],
        "u_magnitude": u_magnitude,
    }

    # Save the fields in VTK format
    save_fields_vtk(fields, timestep=step)

    # Save the u_magnitude slice at the mid y-plane
    mid_y = grid_shape[1] // 2
    save_image(fields["u_magnitude"][:, mid_y, :], timestep=step)

    # Save monitor plot
    plot_uplus(u, step, grid_shape, u_tau, visc)


# Plotting Function
def plot_uplus(u, timestep, grid_shape, u_tau, visc):
    # Mean streamwise velocity in wall units u^+(z)
    zz = np.arange(grid_shape[-1])
    zz = np.minimum(zz, zz.max() - zz)
    yplus = zz * u_tau / visc
    uplus = np.mean(u[0], axis=(0, 1)) / u_tau
    uplus_loglaw = vonKarman_loglaw_wall(yplus)
    dns_dic = get_dns_data()

    plt.clf()
    plt.semilogx(yplus, uplus, "r.", label="Simulation")
    plt.semilogx(yplus, uplus_loglaw, "k:", label="Von Karman Log Law")
    plt.semilogx(dns_dic["y+"], dns_dic["Umean"], "b-", label="DNS Data")
    ax = plt.gca()
    ax.set_xlim([0.1, 300])
    ax.set_ylim([0, 20])
    plt.xlabel("y+")
    plt.ylabel("U+")
    plt.title(f"u+ vs y+ at timestep {timestep}")
    plt.legend()
    fname = f"uplus_{str(timestep).zfill(5)}.png"
    plt.savefig(fname, format="png")
    plt.close()


# -------------------------- Simulation Loop --------------------------

start_time = time.time()
for step in range(num_steps):
    f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
    f_0, f_1 = f_1, f_0  # Swap the buffers

    if (step + 1) % print_interval == 0:
        elapsed_time = time.time() - start_time
        print(f"Iteration: {step + 1}/{num_steps} | Time elapsed: {elapsed_time:.2f}s")
        start_time = time.time()

    if (step % post_process_interval == 0) or (step == num_steps - 1):
        post_process(step, f_0, grid_shape, macro)

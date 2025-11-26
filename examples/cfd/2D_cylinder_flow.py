import jax
import jax.numpy as jnp
import numpy as np
import xlb
from xlb import ComputeBackend, PrecisionPolicy
from xlb.velocity_set.d2q9 import D2Q9
from xlb.grid import grid_factory
from xlb.operator.collision.bgk import BGK
from xlb.operator.equilibrium.quadratic_equilibrium import QuadraticEquilibrium
from xlb.operator.boundary_condition import (
    ZouHeBC,
    FullwayBounceBackBC,
)
from xlb.operator.stream import Stream
from xlb.operator.macroscopic import Macroscopic
from xlb.helper.nse_solver import create_nse_fields

# --- Configuration ---
# Physical parameters
Re = 100.0
u_max = 0.1
cylinder_radius = 10
L_y = 100
nu = u_max * (2 * cylinder_radius) / Re
tau = 3.0 * nu + 0.5
omega = 1.0 / tau
print(f"Re: {Re}, u_max: {u_max}, nu: {nu}, tau: {tau}, omega: {omega}")

# Grid parameters
nx, ny = 800, 200

# Backend and Precision
compute_backend = ComputeBackend.JAX
precision_policy = PrecisionPolicy.FP32FP32

# Initialize Velocity Set
velocity_set = D2Q9(precision_policy, compute_backend)

# Initialize XLB
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# --- Setup ---
# Create Grid
grid = grid_factory((nx, ny))

# Create Fields
grid, f_0, f_1, missing_mask, bc_mask = create_nse_fields(grid=grid)

# Define Geometry (Cylinder)
# Coordinate arrays
x_coords = jnp.arange(nx)
y_coords = jnp.arange(ny)
X, Y = jnp.meshgrid(x_coords, y_coords, indexing="ij") # Shape (nx, ny)

cylinder_mask = (X - ny//2)**2 + (Y - ny//2)**2 <= cylinder_radius**2
# Add cylinder to missing mask (solid obstacle)
missing_mask = missing_mask.at[:, cylinder_mask].set(True)


# --- Boundary Conditions ---
# Instantiate BCs first to get IDs
# Use NumPy array for prescribed values as ZouHeBC expects np.ndarray
u_inlet = np.array([u_max, 0.0]) # ux, uy

zouhe_inlet = ZouHeBC(
    bc_type="velocity",
    prescribed_values=u_inlet,
)

zouhe_outlet = ZouHeBC(
    bc_type="pressure",
    prescribed_values=1.0, # Scalar for pressure
)

wall_bc = FullwayBounceBackBC()

cylinder_bc = FullwayBounceBackBC()

# Get IDs
BC_INLET = zouhe_inlet.id
BC_OUTLET = zouhe_outlet.id
BC_WALL = wall_bc.id
BC_CYLINDER = cylinder_bc.id

print(f"BC IDs: Inlet={BC_INLET}, Outlet={BC_OUTLET}, Wall={BC_WALL}, Cylinder={BC_CYLINDER}")

# Set BC Mask
# Inlet (Left, x=0) - all y positions at x=0
bc_mask = bc_mask.at[0, 0, :].set(BC_INLET)
# Outlet (Right, x=nx-1) - all y positions at x=nx-1
bc_mask = bc_mask.at[0, -1, :].set(BC_OUTLET)
# Top Wall (y=ny-1) - all x positions at y=ny-1
bc_mask = bc_mask.at[0, :, -1].set(BC_WALL)
# Bottom Wall (y=0) - all x positions at y=0
bc_mask = bc_mask.at[0, :, 0].set(BC_WALL)
# Cylinder Surface
bc_mask = bc_mask.at[0, cylinder_mask].set(BC_CYLINDER)

# Note: ZouHe BC computes missing_mask internally from bc_mask
# Only the cylinder needs missing_mask set (solid obstacle)

# List of BCs for iteration
bcs = [zouhe_inlet, zouhe_outlet, wall_bc, cylinder_bc]


# --- Operators ---
eq_op = QuadraticEquilibrium()
collision_op = BGK()
stream_op = Stream()
macroscopic_op = Macroscopic()


# --- Simulation Loop ---
# Initialize fields
rho_init = jnp.ones((1, nx, ny))
u_init = jnp.zeros((2, nx, ny))
# Add slight perturbation or initial flow to speed up?
u_init = u_init.at[0, ...].set(u_max)

# Equilibrium initialization
f_0 = eq_op(rho_init, u_init)
f_1 = jnp.zeros_like(f_0)

# JIT compile the step
@jax.jit
def step(f_pre, f_post, bc_mask, missing_mask):
    # 1. Macroscopic
    rho, u = macroscopic_op(f_pre)
    
    # 2. Equilibrium
    feq = eq_op(rho, u)
    
    # 3. Collision
    f_out = collision_op(f_pre, feq, rho, u, omega)
    
    # 4. Stream
    f_streamed = stream_op(f_out)
    
    # 5. Boundary Conditions
    # Apply BCs sequentially
    f_curr = f_streamed
    for bc in bcs:
        f_curr = bc(f_streamed, f_curr, bc_mask, missing_mask)
    
    return f_curr

# Run
num_steps = 20000
save_interval = 10  # Save every 10 steps
print("Starting simulation...")
import time
start_time = time.time()

# Storage for visualization
saved_steps = []
saved_rho = []
saved_u = []
saved_vorticity = []

# We need to swap f_0 and f_1 buffers
current_f = f_0
next_f = f_1

for i in range(num_steps):
    next_f = step(current_f, next_f, bc_mask, missing_mask)
    
    # Swap
    current_f, next_f = next_f, current_f
    
    # Save data for visualization
    if i % save_interval == 0:
        rho, u = macroscopic_op(current_f)
        # Compute vorticity (curl of velocity in 2D)
        # vorticity = du_y/dx - du_x/dy
        du_y_dx = jnp.gradient(u[1], axis=0)
        du_x_dy = jnp.gradient(u[0], axis=1)
        vorticity = du_y_dx - du_x_dy
        
        saved_steps.append(i)
        saved_rho.append(np.array(rho[0]))
        saved_u.append(np.array(u))
        saved_vorticity.append(np.array(vorticity))
    
    if i % 100 == 0:
        rho, u = macroscopic_op(current_f)
        u_mag = jnp.sqrt(u[0]**2 + u[1]**2)
        print(f"Step {i}: Max U = {jnp.max(u_mag):.4f}, Min Rho = {jnp.min(rho):.4f}")

end_time = time.time()
print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
print(f"MLUPS: {num_steps * nx * ny / (end_time - start_time) / 1e6:.2f}")

# Convert to numpy arrays for easier handling
saved_rho = np.array(saved_rho)
saved_u = np.array(saved_u)
saved_vorticity = np.array(saved_vorticity)

print(f"\nSaved {len(saved_steps)} frames for visualization")

# Save data to files for visualization
np.save('saved_steps.npy', np.array(saved_steps))
np.save('saved_rho.npy', saved_rho)
np.save('saved_u.npy', saved_u)
np.save('saved_vorticity.npy', saved_vorticity)
print("Data saved to .npy files")
print("\nRun 'python plot_flow.py' to visualize the results")

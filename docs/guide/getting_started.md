# Getting Started: Your First Simulation

Welcome to XLB! This guide will walk you through every step of creating and running your first 2D fluid simulation: the classic **Lid-Driven Cavity**.

By the end of this tutorial, you will understand the workflow to simulate fluid in a box with a moving top lid and know how the results are generated.

## The Goal: Lid-Driven Cavity

We will simulate a square cavity filled with a fluid. The bottom and side walls are stationary, while the top wall (the "lid") moves at a constant horizontal velocity. This motion drags the fluid, creating a large vortex inside the cavity.


---

### Step 1: Imports and Initial Configuration

First, we need to import the necessary components from the `xlb` library and other utilities. We also define our simulation's core configuration:

* **`ComputeBackend`**: We choose the engine that will perform the calculations. `WARP` is a great choice for high performance on NVIDIA GPUs.
* **`PrecisionPolicy`**: We define the numerical precision for our calculations (e.g., 32-bit floats).
* **`velocity_set`**: We choose the set of discrete velocities for our lattice. For 2D simulations, `D2Q9` is the standard choice.

```python
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import HalfwayBounceBackBC, EquilibriumBC
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_image

import jax.numpy as jnp
import numpy as np
import warp as wp
```

### Step 2: Define Simulation Parameters

Next, we set the physical and numerical parameters for our simulation in the script.

* **Grid Shape**: We'll use a 500x500 grid.
* **Reynolds Number (`Re`)**: A dimensionless number that characterizes the flow. Higher `Re` means more turbulent-like flow.
* **Lid Velocity (`u_lid`)**: The speed of the top wall.
* **Relaxation Parameter (`omega`)**: This parameter, related to the fluid's viscosity, controls the rate at which particle distributions relax to equilibrium during the collision step. It is calculated from the Reynolds number. Its value is typically between 0 and 2.

```python
# --- Simulation Parameters ---
GRID_SHAPE = (500, 500)
REYNOLDS_NUMBER = 200.0
LID_VELOCITY = 0.05
NUM_STEPS = 10000

# --- Derived Parameters ---
# Calculate fluid viscosity from Reynolds number
viscosity = LID_VELOCITY * (GRID_SHAPE[0] - 1) / REYNOLDS_NUMBER
# Calculate the relaxation parameter omega from viscosity
omega = 1.0 / (3.0 * viscosity + 0.5)
```

### Step 3: Initialize XLB and Create the Grid

With our configuration ready, we initialize the XLB environment and create our simulation grid.

* `xlb.init()`: This crucial step sets up the global environment with our chosen backend and precision.
* `grid_factory()`: This function creates a `Grid` object, which is a representation of our simulation domain.

```python
# --- Setup XLB Environment ---
compute_backend = ComputeBackend.WARP
precision_policy = PrecisionPolicy.FP32FP32
velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=compute_backend)

# Initialize XLB
xlb.init(
    velocity_set=velocity_set,
    default_backend=compute_backend,
    default_precision_policy=precision_policy,
)

# Create the simulation grid
grid = grid_factory(GRID_SHAPE, compute_backend=compute_backend)
```

### Step 4: Define Boundary Regions

We need to tell XLB where the walls and the moving lid are. The `grid` object has a helper method, `bounding_box_indices()`, to easily get the indices of the domain's edges, which we then assign to named regions like `lid` and `walls`.

```python
# --- Define Boundary Indices ---
# Get all boundary indices
box = grid.bounding_box_indices()
# Get boundary indices without the corners
box_no_edge = grid.bounding_box_indices(remove_edges=True)

# Assign regions
lid = box_no_edge["top"]
walls = [box["bottom"][i] + box["left"][i] + box["right"][i] for i in range(velocity_set.d)]
# Flatten the list of wall indices
walls = np.unique(np.array(walls), axis=-1).tolist()
```

### Step 5: Setup Boundary Conditions

Now we associate a physical behavior with each boundary region by creating instances of boundary condition classes.

* **`EquilibriumBC`**: We use this for the lid. It forces the fluid at the lid's location to have a specific density $\rho$ and velocity $\vec{u}$. This effectively "drags" the fluid.
* **`HalfwayBounceBackBC`**: This is a standard no-slip wall condition. It simulates a solid, stationary wall by bouncing particles back in the direction they came from.

```python
# --- Create Boundary Conditions ---
bc_lid = EquilibriumBC(rho=1.0, u=(LID_VELOCITY, 0.0), indices=lid)
bc_walls = HalfwayBounceBackBC(indices=walls)

boundary_conditions = [bc_walls, bc_lid]
```

### Step 6: Setup the Stepper

The `Stepper` is the engine of the simulation. It orchestrates the collision and streaming steps. We create an `IncompressibleNavierStokesStepper` and use its `prepare_fields()` method to initialize the distribution function arrays (`f_0` and `f_1`) and boundary masks.

`f_0` and `f_1` are two buffers that hold the entire state of the simulation. We use two to swap between them at each time step.

```python
# --- Setup the Simulation Stepper ---
stepper = IncompressibleNavierStokesStepper(
    grid=grid,
    boundary_conditions=boundary_conditions,
    collision_type="BGK",
)

# Prepare the fields (distribution functions f_0, f_1 and masks)
f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()
```

### Step 7: The Simulation Loop

This is where the magic happens! A `for` loop runs the simulation for a specified number of steps. In each iteration of the loop, the script performs these actions:
1.  Calls the `stepper` to perform one full time step (collision and streaming). This updates the `f_1` buffer based on the current state in the `f_0` buffer.
2.  **Swaps the buffers**: The script then swaps `f_0` and `f_1` so that the newly computed state becomes the input for the next step.
3.  Periodically runs post-processing logic to calculate macroscopic variables like velocity and save the results as an image.

```python
# --- Run the Simulation ---
print("Starting simulation...")
for step in range(NUM_STEPS):
    # Perform one simulation step
    f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)

    # Swap the distribution function buffers
    f_0, f_1 = f_1, f_0

    # --- Post-processing (every 1000 steps) ---
    if step % 1000 == 0 or step == NUM_STEPS - 1:
        print(f"Processing step {step}...")
        
        # We use a JAX-backend Macroscopic computer for post-processing
        # First, convert the Warp tensor to a JAX array if needed
        if compute_backend == ComputeBackend.WARP:
            f_current = wp.to_jax(f_0)[..., 0] # Drop the z-dim added by Warp for 2D
        else:
            f_current = f_0

        # Create a Macroscopic computer on the fly
        macro_computer = Macroscopic(
            compute_backend=ComputeBackend.JAX,
            precision_policy=precision_policy,
            velocity_set=xlb.velocity_set.D2Q9(precision_policy=precision_policy, compute_backend=ComputeBackend.JAX),
        )
        
        # Calculate density and velocity
        rho, u = macro_computer(f_current)
        
        # Calculate velocity magnitude for visualization
        u_magnitude = jnp.sqrt(u[0]**2 + u[1]**2)
        
        # Save the velocity magnitude field as an image
        save_image(u_magnitude, timestep=step, prefix="lid_driven_cavity")
        print(f"Saved image for step {step}.")

print("Simulation finished!")
```

## Next Steps

Congratulations! You now understand the workflow for configuring and running a fluid simulation with XLB.
* To understand the details of the classes and functions we discussed, dive into the [**API Reference**](../api/constants.md).
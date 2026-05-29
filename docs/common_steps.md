# 🔁 Common Flow / Required Steps Across All Examples

## **Step 1: Import and Configure Environment**
- Import `xlb` components.
- Define backend (`WARP` or `JAX`) and precision policy.
- Choose appropriate `velocity_set` based on simulation type (2D or 3D).

## **Step 2: Initialize XLB**
```python
xlb.init(velocity_set, default_backend, default_precision_policy)
```

## **Step 3: Define the Simulation Grid**
```python
grid = grid_factory(grid_shape, compute_backend)
```

## **Step 4: Define Boundary Indices**
```python
box = grid.bounding_box_indices()
walls = ...
```

## **Step 5: Setup Boundary Conditions**
```python
bc_wall = FullwayBounceBackBC(...)
bc_inlet = RegularizedBC(...)
boundary_conditions = [bc_wall, bc_inlet, ...]
```

## **Step 6: Setup Stepper**
```python
stepper = IncompressibleNavierStokesStepper(grid, boundary_conditions, ...)
f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()
```

## **Step 7 (Optional): Initialize Fields**
```python
f_0 = initialize_eq(f_0, grid, velocity_set, precision_policy, compute_backend, u=u_init)
```

## **Step 8: Post-processing Setup**
```python
Instantiate Macroscopic.
Write custom post_process functions.
```

## **Step 9: Run Simulation Loop**
```python
for step in range(num_steps):
    f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
    f_0, f_1 = f_1, f_0  # Swap buffers
    if step % interval == 0:
        post_process(...)
```
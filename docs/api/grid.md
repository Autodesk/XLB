# Grid

The `xlb.grid` module provides the fundamental tools for defining and managing the structured, Cartesian grids used in Lattice Boltzmann simulations. A `Grid` object represents the spatial layout of your simulation domain and is the first component you typically create when setting up a simulation.

It is tightly integrated with the selected compute backend, ensuring that all data structures are allocated on the correct device (e.g., a JAX device or an NVIDIA GPU for Warp).

---

## Creating a Grid

The primary way to create a grid is with the `grid_factory` function. This function automatically returns the correct grid object (`JaxGrid` or `WarpGrid`) based on the selected compute backend.

```python
from xlb.grid import grid_factory
from xlb.compute_backend import ComputeBackend
import xlb

# Assuming xlb.init() has been called with a default backend...

# Create a 2D grid for the JAX backend
grid_2d = grid_factory(shape=(500, 500), compute_backend=ComputeBackend.JAX)

# Create a 3D grid for the Warp backend
grid_3d = grid_factory(shape=(256, 128, 128), compute_backend=ComputeBackend.WARP)
```

### `grid_factory(shape, compute_backend)`
- **`shape: Tuple[int, ...]`**: A tuple defining the grid dimensions. For example, `(nx, ny)` for 2D or `(nx, ny, nz)` for 3D.
- **`compute_backend: ComputeBackend`**: The backend to use (`ComputeBackend.JAX` or `ComputeBackend.WARP`). If not provided, it defaults to the backend set in `xlb.init()`.


---

## Using a Grid Instance

Once you have a `grid` object, you can use its attributes and methods to define boundary regions and create data fields for your simulation.

### Attributes

- **`grid.shape`**: The full domain shape passed during creation (e.g., `(256, 128, 128)`).
- **`grid.dim`**: The number of spatial dimensions, inferred from the shape (2 for 2D, 3 for 3D).

### Methods

#### `grid.bounding_box_indices()`

This is a crucial helper method for defining boundary conditions. It returns a dictionary containing the integer coordinates for each face of the grid's bounding box.

```python
faces = grid_2d.bounding_box_indices(remove_edges=True)

# faces is a dictionary with keys: 'bottom', 'top', 'left', 'right'
inlet_indices = faces["left"]
outlet_indices = faces["right"]

# Combine multiple faces to define all stationary walls
wall_indices = faces["bottom"] + faces["top"]
```

- **`remove_edges: bool = False`**: If set to `True`, the corner/edge nodes where faces meet are excluded. This is highly recommended to prevent applying conflicting boundary conditions to the same node (e.g., treating a corner as both a "left" wall and a "bottom" wall).

#### `grid.create_field()`

This method allocates a data array (a "field") on the grid, using the appropriate backend (e.g., `jax.numpy` array or `warp` array). This is used to create storage for all simulation data, such as the particle distribution functions ($f_i$) or macroscopic quantities ($\rho, \vec{u}$).

```python
# Create a field to store the D2Q9 particle distribution functions (f_i)
# Cardinality is 9 because there are 9 velocities in D2Q9.
f = grid_2d.create_field(cardinality=9)

# Create a scalar field for density (rho)
rho = grid_2d.create_field(cardinality=1)

# Create a 2D vector field for velocity (u)
u = grid_2d.create_field(cardinality=2)
```

- **`cardinality: int`**: The number of data values to store at each grid node. For a scalar field like density, `cardinality=1`. For a vector field like 2D velocity, `cardinality=2`. For the LBM particle populations $f_i$, `cardinality` is equal to $q$, the number of discrete velocities in your `VelocitySet`.
- **`dtype: Precision = None`**: The numerical precision for the field data (e.g., `Precision.FP32`). Defaults to the global precision policy. `Precision.BOOL` is only supported by `JaxGrid`.
- **`fill_value: float = None`**: An optional value to initialize all elements of the array. Defaults to `0`.

---

## Backend-Specific Details

### `JaxGrid`
The `JaxGrid` object is designed to be compatible with JAX's features, including multi-device parallelization via sharding.

### `WarpGrid`
The `WarpGrid` is optimized for single-GPU execution with NVIDIA Warp. For 2D grids, it automatically adds a singleton `z` dimension (e.g., a `(500, 500)` shape becomes a `(500, 500, 1)` array). This is done to maintain consistency, allowing the same Warp kernels to be used for both 2D and 3D simulations.
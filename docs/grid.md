# Grid

The `xlb.grid` module provides utilities to define and manage structured grids for Lattice Boltzmann (LBM) simulations. A grid defines the spatial layout of your simulation domain and is tightly integrated with the selected compute backend (JAX or Warp).

## Quick start

```python
from xlb.grid import grid_factory
from xlb.compute_backend import ComputeBackend
import xlb

# (Usually already done once in your program)
xlb.init(...)

# Create a 3D grid on Warp
grid = grid_factory((256, 128, 128), compute_backend=ComputeBackend.WARP)

# Create a distribution field with cardinality = velocity_set.d
f = grid.create_field(cardinality=19)  # e.g., D3Q19

# Get boundary indices to build BCs
faces = grid.bounding_box_indices(remove_edges=True)
left, right = faces["left"], faces["right"]

```

## **Parameters**

- `shape: Tuple[int, ...]` — `(nx, ny)` for 2D or `(nx, ny, nz)` for 3D.
- `compute_backend: ComputeBackend` — `ComputeBackend.JAX` or `ComputeBackend.WARP`.

## **Attributes**

- `grid.shape` : The full domain shape you passed in.
- `grid.dim` : 2 or 3, inferred from `shape`.

## **Functions**

- `grid_factory(shape, compute_backend=None)` : Returns a `JaxGrid` or `WarpGrid` based on the backend.

- `grid.bounding_box_indices(remove_edges: bool = False) -> dict[str, list[list[int]]]`:
    - Returns integer indices for each boundary face.
    - Keys for 2D: `bottom`, `top`, `left`, `right`
    - Keys for 3D: `bottom`, `top`, `left`, `right`, `front`, `back`
    - `remove_edges=True` : removes edge/corner nodes (useful to avoid double-applying BCs).

Example
```python
faces = grid.bounding_box_indices(remove_edges=True)
inlet  = faces["left"]
outlet = faces["right"]
walls  = (faces["bottom"][0] + faces["top"][0])  # merge faces per-dimension
```


- `grid.create_field(cardinality: int, dtype: Precision = None, fill_value: float = None) -> array`:
    - Creates a field over the grid for storing simulation data.
    - `cardinality`: number of values per grid cell
        - 1 → scalar field (e.g., density, pressure)
        - 2/3 → vector field (e.g., velocity components)
    - `dtype`: precision (FP32, FP64, FP16, BOOL). Defaults to config.
        - `BOOL` is only supported with **JaxGrid**
    - `fill_value`: initialize with constant; defaults to 0.



## **JAX-specific:** `JaxGrid`
`JaxGrid` is designed for multi-device setups.


## **Warp-specific:** `WarpGrid`
`WarpGrid` targets single GPU with NVIDIA Warp.

- For 2D grids, a singleton `z` dimension is automatically added to keep kernels consistent.


# 📘 Constants and Enums

This page documents the core enums and configuration constants used throughout the codebase. These constants define important fixed values such as compute backends, grid strategies, physics models, and precision policies used in simulations.



## 🔧 Compute Backends

Defined in compute_backend.py

```python
class ComputeBackend(Enum):
    JAX = auto()
    WARP = auto()
```
Description:

Specifies the available compute engines:

- **JAX**: Uses JAX for GPU/TPU accelerated computation.
- **WARP**: Uses NVIDIA Warp for GPU-based simulation.

## 🧱 Grid Backends

Defined in grid_backend.py
```python
class GridBackend(Enum):
    JAX = auto()
    WARP = auto()
    OOC = auto()
```

Description:

Represents the grid computation backend:

- **JAX, WARP** (same as above)
- **OOC**: Out-of-core grid handling (e.g., large datasets or disk-based grids).

## 🌊 Physics Types

Defined in physics_type.py
```python
class PhysicsType(Enum):
    NSE = auto()  # Navier-Stokes Equations
    ADE = auto()  # Advection-Diffusion Equations
```

Description:

Defines the physical equations the system can solve:

- **NSE**: Fluid dynamics using Navier-Stokes.
- **ADE**: Transport processes using Advection-Diffusion.


## 🎯 Precision Enum

Defined in precision_policy.py
```python
class Precision(Enum):
    FP64, FP32, FP16, UINT8, BOOL
```

Description:

Represents data precision levels. Each precision level maps to both JAX and WARP data types via properties:

- Precision.wp_dtype → Warp data type
- Precision.jax_dtype → JAX data type



## ⚙️ PrecisionPolicy Enum

```python
class PrecisionPolicy(Enum):
    FP64FP64, FP64FP32, FP64FP16, FP32FP32, FP32FP16
```

Description:

Controls how data is computed vs stored:

- **FP64FP32** means compute in float64, store in float32
Utility methods:
- **.compute_precision**: Returns compute-side precision
- **.store_precision**: Returns storage-side precision
- **.cast_to_compute_jax(array)**: Casts to compute dtype
- **.cast_to_store_jax(array)**: Casts to store dtype



## ⚙️ Default Configuration

Defined in default_config.py

```python
@dataclass
class DefaultConfig:
    velocity_set
    default_backend
    default_precision_policy
```
Set globally using:
```python
init(velocity_set, backend, precision_policy)
```
This is used to initialize system-wide simulation behavior based on chosen backends and numerical settings.
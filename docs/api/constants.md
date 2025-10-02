# Constants and Enums

This page provides a reference for the core enumerations (`Enum`) and configuration objects that govern the behavior of XLB simulations. These objects are used to specify settings like the computational backend, numerical precision, and the physics model to be solved.

---

## ComputeBackend

Defined in `compute_backend.py`

```python
class ComputeBackend(Enum):
    JAX = auto()
    WARP = auto()
```

**Description:**

An `Enum` specifying the primary computational engine for executing simulation kernels.

- **`JAX`**: Use the [JAX](https://github.com/google/jax) framework for computation, enabling execution on CPUs, GPUs, and TPUs.
- **`WARP`**: Use the [NVIDIA Warp](https://github.com/NVIDIA/warp) framework for high-performance GPU simulation kernels.

---

## GridBackend

Defined in `grid_backend.py`

```python
class GridBackend(Enum):
    JAX = auto()
    WARP = auto()
    OOC = auto()
```

**Description:**

An `Enum` defining the backend for grid creation and data management.

- **`JAX`**, **`WARP`**: The grid data resides in memory on the respective compute device.
- **`OOC`**: Handles simulations where the grid data is too large to fit into memory and must be processed "out-of-core" from disk.

---

## PhysicsType

Defined in `physics_type.py`

```python
class PhysicsType(Enum):
    NSE = auto()  # Navier-Stokes Equations
    ADE = auto()  # Advection-Diffusion Equations
```

**Description:**

An `Enum` used to select the set of physical equations to be solved by the stepper.

- **`NSE`**: Simulates fluid dynamics governed by the incompressible Navier-Stokes equations.
- **`ADE`**: Simulates transport phenomena governed by the Advection-Diffusion equation.

---

## Precision

Defined in `precision_policy.py`

```python
class Precision(Enum):
    FP64 = auto()
    FP32 = auto()
    FP16 = auto()
    UINT8 = auto()
    BOOL = auto()
```

**Description:**

An `Enum` representing fundamental data precision levels. Each member provides properties to get the corresponding data type in the target compute backend:

- **`.wp_dtype`**: The equivalent `warp` data type (e.g., `wp.float32`).
- **`.jax_dtype`**: The equivalent `jax.numpy` data type (e.g., `jnp.float32`).

---

## PrecisionPolicy

Defined in `precision_policy.py`

```python
class PrecisionPolicy(Enum):
    FP64FP64 = auto()
    FP64FP32 = auto()
    FP64FP16 = auto()
    FP32FP32 = auto()
    FP32FP16 = auto()
```

**Description:**

An `Enum` that defines a policy for balancing numerical accuracy and memory usage. It specifies a precision for computation and a (potentially different) precision for storage.

For example, `FP64FP32` specifies that calculations should be performed in high-precision `float64`, but the results are stored in memory-efficient `float32`.

**Utility Properties & Methods:**
- **`.compute_precision`**: Returns the `Precision` enum for computation.
- **`.store_precision`**: Returns the `Precision` enum for storage.
- **`.cast_to_compute_jax(array)`**: Casts a JAX array to the policy's compute precision.
- **`.cast_to_store_jax(array)`**: Casts a JAX array to the policy's store precision.

---

## DefaultConfig

Defined in `default_config.py`

```python
@dataclass
class DefaultConfig:
    velocity_set
    default_backend
    default_precision_policy
```

A `dataclass` that holds the global configuration for a simulation session.

An instance of this configuration is set globally using the `xlb.init()` function at the beginning of a script. This ensures that all subsequently created XLB components are aware of the chosen backend, velocity set, and precision policy.

```python
# The xlb.init() function sets the global DefaultConfig instance
xlb.init(
    velocity_set=D2Q9(...),
    default_backend=ComputeBackend.JAX,
    default_precision_policy=PrecisionPolicy.FP32FP32
)
```
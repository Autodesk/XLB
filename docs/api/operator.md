# The Operator Concept

In XLB, an **Operator** is a fundamental building block that performs a single, well-defined action within a simulation. Think of operators as the "verbs" of the Lattice Boltzmann Method. Any distinct step in the algorithm, such as collision, streaming, or calculating equilibrium, is encapsulated within its own callable operator object.

---

## Common Features (The Operator Base Class)

The `Operator` base class provides a unified interface and a set of shared features for all operators in XLB. This ensures that every part of the simulation behaves consistently, regardless of its specific function. When you use any operator in XLB, you can rely on the following features:

#### 1. Backend Management
Every operator is aware of the compute backend (`JAX` or `Warp`). When an operator is called, it automatically dispatches the execution to the correct, highly optimized backend implementation without any extra effort from the user.

#### 2. Precision Policy Awareness
Operators automatically respect the globally or locally defined `PrecisionPolicy`. They handle the data types for computation (`.compute_dtype`) and storage (`.store_dtype`) transparently, helping to balance performance and numerical accuracy.

#### 3. Standardized Calling Convention
All operators are **callable** (using `()`). This provides a clean, functional API that makes it simple to compose operators and build a custom simulation loop.

### Conceptual Usage

While you rarely instantiate the base `Operator` directly, all its subclasses follow the same pattern of creation and use. You first instantiate a specific operator (e.g., for BGK collision), configuring it with a velocity set and policies. Then, you call that instance with the required data to perform its action.

```python
from xlb.operator.collision import BGKCollision # A concrete operator subclass
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend

# Instantiate a specific operator (e.g., for BGK collision)
collision_op = BGKCollision(
    velocity_set=my_velocity_set,
    precision_policy=PrecisionPolicy.FP32FP32,
    compute_backend=ComputeBackend.JAX
)

# Call the operator with the required data
# It automatically runs the JAX implementation in FP32.
f_post_collision = collision_op(f_pre_collision, omega=1.8)
```

---

## Utility Operator: `PrecisionCaster`

The `PrecisionCaster` is a prime example of a simple yet powerful utility operator provided by XLB. Its sole purpose is to convert the numerical precision of simulation data fields.

### Overview

Precision plays an important role in balancing **accuracy** and **performance**. Some simulation steps may require high precision (`float64`) for stability, while many others can run much faster in lower precision (`float32`), especially on GPUs. The `PrecisionCaster` handles this conversion seamlessly.

### Use Cases

- **Performance Optimization**: Run the bulk of a simulation in `FP32` for speed, but cast data to `FP64` before critical calculations.
- **Mixed-Precision Workflows**: Adapt precision dynamically based on runtime stability needs.
- **Memory Management**: Store data in a lower precision to reduce memory footprint, casting to a higher precision only when needed for computation.

### Usage
To use the `PrecisionCaster`, you create an instance configured with the *target* precision policy you want to convert to. You then apply this operator to a data field, and it will return a new field with the converted precision.

```python
from xlb.operator import PrecisionCaster

# Assume 'f_low_precision' is a field with FP32 data
# Create a caster to convert data to a higher precision (FP64)
caster_to_fp64 = PrecisionCaster(
    velocity_set=my_velocity_set,
    precision_policy=PrecisionPolicy.FP64FP64 # Target policy
)

# Apply the operator to cast the field
f_high_precision = caster_to_fp64(f_low_precision)
```

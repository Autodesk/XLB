# Operator

Base class for all operators, including collision, streaming, equilibrium, etc.  
Responsible for handling compute backends like JAX and NVIDIA Warp.

## Overview

The `Operator` class acts as the foundational interface for all lattice Boltzmann operators in XLB. It manages backend selection and provides a unified API to call backend-specific implementations transparently. It also facilitates registering backend implementations and handles precision policies and compute types.

## Usage

```python
from xlb.operator import Operator
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy

op = Operator(
    velocity_set=None,  # or specify velocity set
    precision_policy=PrecisionPolicy.FP32FP32,
    compute_backend=ComputeBackend.JAX,
)

# Call the operator (calls backend-specific implementation)
result = op(some_input_data)
```

## Constructor

```python
Operator(
    velocity_set=None,
    precision_policy=None,
    compute_backend=None
)
```

- velocity_set: (optional) Velocity set used by the operator; defaults to global config.
- precision_policy: (optional) Precision policy for compute and storage.
- compute_backend: (optional) Backend to run on (e.g., JAX or Warp).

Raises `ValueError` if the specified backend is unsupported.

## Methods and Properties

`register_backend(backend_name)`
Decorator to register backend implementations for subclasses.

`__call__(*args, callback=None, **kwargs)`
Calls the appropriate backend method, matching the subclass and backend, and passes args/kwargs.
- callback (optional): Callable to be called with the result.

`supported_compute_backend`
Returns a list of supported backend keys registered for this operator.

`backend`
Returns the actual backend module (jax.numpy or warp), depending on the current backend.

`compute_dtype`
Returns the compute data type (e.g., float32, float64) according to the precision policy and backend.

`store_dtype`
Returns the storage data type according to the precision policy and backend.
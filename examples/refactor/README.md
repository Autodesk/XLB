# Refactor Examples

This directory contains several example of using the refactored XLB library.

These examples are not meant to be veiwed as the new interface to XLB but only how
to expose the compute kernels to a user. Development is still ongoing.

## Examples

### JAX Example

The JAX example is a simple example of using the refactored XLB library
with JAX. The example is located in the `example_jax.py`. It shows
a very basic flow past a cyliner.

### NUMBA Example

TODO: Not working yet

The NUMBA example is a simple example of using the refactored XLB library
with NUMBA. The example is located in the `example_numba.py`. It shows
a very basic flow past a cyliner. This example is not working yet though and
is still under development for numba backend.

### Out of Core JAX Example

This shoes how we can use out of core memory with JAX. The example is located
in the `example_jax_out_of_core.py`. It shows a very basic flow past a cyliner.
The basic idea is to create an out of core memory array using the implementation
in XLB. Then we run the simulation using the jax functions implementation obtained 
from XLB. Some rendering is done using PhantomGaze.

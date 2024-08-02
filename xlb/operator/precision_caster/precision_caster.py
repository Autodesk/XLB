"""
Base class for casting precision of the input data to the desired precision
"""

import jax.numpy as jnp
from jax import jit
import warp as wp
from functools import partial

from xlb.operator.operator import Operator
from xlb.velocity_set import VelocitySet
from xlb.precision_policy import Precision, PrecisionPolicy
from xlb.compute_backend import ComputeBackend


class PrecisionCaster(Operator):
    """
    Class that handles the construction of lattice boltzmann precision casting operator.
    """

    def __init__(
        self,
        input_precision: Precision,
        output_precision: Precision,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend,
    ):
        super().__init__(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )

        # Set the input and output precision based on the backend
        self.input_precision = self._precision_to_dtype(input_precision)
        self.output_precision = self._precision_to_dtype(output_precision)

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(self, f: jnp.ndarray) -> jnp.ndarray:
        return self.output_precision(f)

    def _construct_warp(self):
        # Construct needed types and constants
        from_lattice_vec = wp.vec(self.velocity_set.q, dtype=self.input_precision)
        to_lattice_vec = wp.vec(self.velocity_set.q, dtype=self.output_precision)
        from_array_type = wp.array4d(dtype=self.input_precision)
        to_array_type = wp.array4d(dtype=self.output_precision)
        _q = wp.constant(self.velocity_set.q)

        # Construct the functional
        @wp.func
        def functional(
            from_f: from_lattice_vec,
        ) -> to_lattice_vec:
            to_f = to_lattice_vec()
            for i in range(self.velocity_set.q):
                to_f[i] = self.output_precision(from_f[i])
            return to_f

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            from_f: from_array_type,
            to_f: to_array_type,
        ):
            # Get the global index
            i, j, k = wp.tid()

            # Get f
            _from_f = from_lattice_vec()
            for l in range(_q):
                _from_f[l] = from_f[l, i, j, k]

            # Cast the precision
            _to_f = functional(_from_f)

            # Set f
            for l in range(_q):
                to_f[l, i, j, k] = _to_f[l]

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, from_f, to_f):
        # Launch the warp kernel
        wp.launch(
            self._kernel,
            inputs=[
                from_f,
                to_f,
            ],
            dim=from_f.shape[1:],
        )
        return to_f

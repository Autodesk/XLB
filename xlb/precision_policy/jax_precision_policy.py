from xlb.precision_policy.base_precision_policy import PrecisionPolicy
from jax import jit
from functools import partial
import jax.numpy as jnp


class JaxPrecisionPolicy(PrecisionPolicy):
    """
    JAX-specific precision policy.
    """

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def cast_to_compute(self, array):
        return array.astype(self.compute_dtype)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def cast_to_store(self, array):
        return array.astype(self.storage_dtype)


class JaxFp32Fp32(JaxPrecisionPolicy):
    """
    Precision policy for lattice Boltzmann method with computation and storage
    precision both set to float32.

    Parameters
    ----------
    None
    """

    def __init__(self):
        super().__init__(jnp.float32, jnp.float32)


class JaxFp64Fp64(JaxPrecisionPolicy):
    """
    Precision policy for lattice Boltzmann method with computation and storage
    precision both set to float64.
    """

    def __init__(self):
        super().__init__(jnp.float64, jnp.float64)


class JaxFp64Fp32(JaxPrecisionPolicy):
    """
    Precision policy for lattice Boltzmann method with computation precision
    set to float64 and storage precision set to float32.
    """

    def __init__(self):
        super().__init__(jnp.float64, jnp.float32)


class JaxFp64Fp16(JaxPrecisionPolicy):
    """
    Precision policy for lattice Boltzmann method with computation precision
    set to float64 and storage precision set to float16.
    """

    def __init__(self):
        super().__init__(jnp.float64, jnp.float16)


class JaxFp32Fp16(JaxPrecisionPolicy):
    """
    Precision policy for lattice Boltzmann method with computation precision
    set to float32 and storage precision set to float16.
    """

    def __init__(self):
        super().__init__(jnp.float32, jnp.float16)

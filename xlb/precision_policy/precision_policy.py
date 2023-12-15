# Precision policy for lattice Boltzmann method
# TODO: possibly refctor this to be more general

from functools import partial
import jax.numpy as jnp
from jax import jit
import numba
from numba import cuda


class PrecisionPolicy(object):
    """
    Base class for precision policy in lattice Boltzmann method.
    Basic idea is to allow for storing the lattice in a different precision than the computation.

    Stores dtype in jax but also contains same information for other backends such as numba.

    Parameters
    ----------
    compute_dtype: jax.numpy.dtype
        The precision used for computation.
    storage_dtype: jax.numpy.dtype
        The precision used for storage.
    """

    def __init__(self, compute_dtype, storage_dtype):
        # Store the dtypes (jax)
        self.compute_dtype = compute_dtype
        self.storage_dtype = storage_dtype

        # Get the corresponding numba dtypes
        self.compute_dtype_numba = self._get_numba_dtype(compute_dtype)
        self.storage_dtype_numba = self._get_numba_dtype(storage_dtype)

        # Check that compute dtype is one of the supported dtypes (float16, float32, float64)
        self.supported_compute_dtypes = [jnp.float16, jnp.float32, jnp.float64]
        if self.compute_dtype not in self.supported_compute_dtypes:
            raise ValueError(
                f"Compute dtype {self.compute_dtype} is not supported. Supported dtypes are {self.supported_compute_dtypes}"
            )

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def cast_to_compute_jax(self, array):
        """
        Cast the array to the computation precision

        Parameters
        ----------
        Array: jax.numpy.ndarray
            The array to cast.

        Returns
        -------
        jax.numpy.ndarray
            The casted array
        """
        return array.astype(self.compute_dtype)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def cast_to_store_jax(self, array):
        """
        Cast the array to the storage precision

        Parameters
        ----------
        Array: jax.numpy.ndarray
            The array to cast.

        Returns
        -------
        jax.numpy.ndarray
            The casted array
        """
        return array.astype(self.storage_dtype)

    def cast_to_compute_numba(self):
        """
        Constructs a numba function to cast a value to the computation precision

        Parameters
        ----------
        value: float
            The value to cast.

        Returns
        -------
        float
            The casted value
        """
        return self._cast_to_dtype_numba(self.compute_dtype_numba)

    def cast_to_store_numba(self):
        """
        Constructs a numba function to cast a value to the storage precision

        Parameters
        ----------
        value: float
            The value to cast.

        Returns
        -------
        float
            The casted value
        """
        return self._cast_to_dtype_numba(self.storage_dtype_numba)

    def _cast_to_dytpe_numba(self, dtype):
        """
        Constructs a numba function to cast a value to the computation precision

        Parameters
        ----------
        value: float
            The value to cast.

        Returns
        -------
        float
            The casted value
        """

        @cuda.jit(device=True)
        def cast_to_dtype(value):
            return dtype(value)

    def _get_numba_dtype(self, dtype):
        """
        Get the corresponding numba dtype

        # TODO: Make this more general

        Parameters
        ----------
        dtype: jax.numpy.dtype
            The dtype to convert

        Returns
        -------
        numba.dtype
            The corresponding numba dtype
        """
        if dtype == jnp.float16:
            return numba.float16
        elif dtype == jnp.float32:
            return numba.float32
        elif dtype == jnp.float64:
            return numba.float64
        elif dtype == jnp.int32:
            return numba.int32
        elif dtype == jnp.int64:
            return numba.int64
        elif dtype == jnp.int16:
            return numba.int16
        else:
            raise ValueError(f"Unsupported dtype {dtype}")

    def __repr__(self):
        return f"compute_dtype={self.compute_dtype}/{self.storage_dtype}"

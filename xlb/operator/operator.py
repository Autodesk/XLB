"""
Base operator module for XLB.

Every LBM operator (collision, streaming, equilibrium, boundary condition,
masker, stepper, etc.) inherits from :class:`Operator`.  The class provides:

* **Backend dispatch** — ``__call__`` automatically selects the registered
  implementation for the active compute backend.
* **Precision management** — ``compute_dtype`` and ``store_dtype`` properties
  return the correct type for the active backend and precision policy.
* **Kernel construction hooks** — ``_construct_warp()`` / ``_construct_neon()``
  are called at init time to compile backend-specific kernels and functionals.
"""

import inspect
import traceback
import jax
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb import DefaultConfig
from xlb.precision_policy import PrecisionPolicy


class Operator:
    """
    Base class for all operators, collision, streaming, equilibrium, etc.

    This class is responsible for handling compute backends.
    """

    _backends = {}

    def __init__(self, velocity_set=None, precision_policy=None, compute_backend=None):
        """Initialize the operator.

        Parameters
        ----------
        velocity_set : VelocitySet, optional
            Lattice velocity set.  Defaults to ``DefaultConfig.velocity_set``.
        precision_policy : PrecisionPolicy, optional
            Precision policy.  Defaults to ``DefaultConfig.default_precision_policy``.
        compute_backend : ComputeBackend, optional
            Compute backend.  Defaults to ``DefaultConfig.default_backend``.
        """
        # Set the default values from the global config
        self.velocity_set = velocity_set or DefaultConfig.velocity_set
        self.precision_policy = precision_policy or DefaultConfig.default_precision_policy
        self.compute_backend = compute_backend or DefaultConfig.default_backend

        # Check if the compute compute_backend is supported
        if self.compute_backend not in ComputeBackend:
            raise ValueError(f"Compute_backend {compute_backend} is not supported")

        # Construct read/write functions for the compute backend
        if self.compute_backend in [ComputeBackend.WARP, ComputeBackend.NEON]:
            self.read_field, self.write_field = self._construct_read_write_functions()
            self.read_field_neighbor = self._construct_read_field_neighbor()

        # Construct the kernel based compute_backend functions TODO: Maybe move this to the register or something
        if self.compute_backend == ComputeBackend.WARP:
            self.warp_functional, self.warp_kernel = self._construct_warp()

        if self.compute_backend == ComputeBackend.NEON:
            self.neon_functional, self.neon_container = self._construct_neon()

        # Updating JAX config in case fp64 is requested
        if self.compute_backend == ComputeBackend.JAX and (
            precision_policy == PrecisionPolicy.FP64FP64 or precision_policy == PrecisionPolicy.FP64FP32
        ):
            jax.config.update("jax_enable_x64", True)

    @classmethod
    def register_backend(cls, backend_name):
        """
        Decorator to register a compute_backend for the operator.
        """

        def decorator(func):
            unwrapped = inspect.unwrap(func)
            qualname = unwrapped.__qualname__
            subclass_name = qualname.rsplit(".", 1)[0] if "." in qualname else qualname
            signature = inspect.signature(unwrapped)
            key = (subclass_name, backend_name, str(signature))
            cls._backends[key] = func
            return func

        return decorator

    def __call__(self, *args, callback=None, **kwargs):
        """Dispatch to the registered backend implementation.

        Iterates over all registered implementations for this operator class
        and the active backend, attempts to bind the provided arguments, and
        executes the first matching signature.  An optional *callback* is
        invoked with the result after successful execution.

        Raises
        ------
        NotImplementedError
            If no implementation is registered for the active backend.
        RuntimeError
            If all candidate implementations raise errors (chained from the last exception).
        """
        method_candidates = [
            (key, method) for key, method in self._backends.items() if key[0] == self.__class__.__name__ and key[1] == self.compute_backend
        ]
        if not method_candidates:
            supported = [key for key in self._backends.keys() if key[0] == self.__class__.__name__]
            raise NotImplementedError(
                f"No implementation found for operator {type(self).__qualname__} with backend {self.compute_backend}. "
                f"Available implementations: {supported}"
            )

        last_key = None
        last_method = None
        error = None
        traceback_str = None
        for key, backend_method in method_candidates:
            last_key = key
            last_method = backend_method
            try:
                # This attempts to bind the provided args and kwargs to the compute_backend method's signature
                bound_arguments = inspect.signature(backend_method).bind(self, *args, **kwargs)
                bound_arguments.apply_defaults()  # This fills in any default values
                result = backend_method(self, *args, **kwargs)
                callback_arg = result if result is not None else (args, kwargs)
                if callback and callable(callback):
                    callback(callback_arg)
                return result
            except Exception as e:
                error = e
                traceback_str = traceback.format_exc()
                continue  # This skips to the next candidate if binding fails

        impl_qualname = inspect.unwrap(last_method).__qualname__ if last_method is not None else "unknown"
        registered_class = last_key[0] if last_key is not None else "unknown"
        instance_class = type(self).__qualname__
        if instance_class != registered_class:
            instance_note = f", instance={instance_class}"
        else:
            instance_note = ""
        raise RuntimeError(f"{impl_qualname} failed for backend {self.compute_backend}{instance_note}: {error}\n{traceback_str}") from error

    @property
    def supported_compute_backend(self):
        """
        Returns the supported compute backend for the operator
        """
        return list(self._backends.keys())

    def _is_method_overridden(self, method_name):
        """
        Helper method to check if a method is overridden in a subclass.
        """
        method = getattr(self, method_name, None)
        if method is None:
            return False
        return method.__func__ is not getattr(Operator, method_name, None).__func__

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @property
    def backend(self):
        """
        Returns the compute backend object for the operator (e.g. jax, warp)
        This should be used with caution as all backends may not have the same API.
        """
        if self.compute_backend == ComputeBackend.JAX:
            import jax.numpy as compute_backend
        elif self.compute_backend == ComputeBackend.WARP:
            import warp as compute_backend
        return compute_backend

    @property
    def compute_dtype(self):
        """
        Returns the compute dtype
        """
        if self.compute_backend == ComputeBackend.JAX:
            return self.precision_policy.compute_precision.jax_dtype
        elif self.compute_backend == ComputeBackend.WARP:
            return self.precision_policy.compute_precision.wp_dtype
        elif self.compute_backend == ComputeBackend.NEON:
            return self.precision_policy.compute_precision.wp_dtype

    @property
    def store_dtype(self):
        """
        Returns the store dtype
        """
        if self.compute_backend == ComputeBackend.JAX:
            return self.precision_policy.store_precision.jax_dtype
        elif self.compute_backend == ComputeBackend.WARP:
            return self.precision_policy.store_precision.wp_dtype
        elif self.compute_backend == ComputeBackend.NEON:
            return self.precision_policy.store_precision.wp_dtype

    def get_precision_policy(self):
        """
        Returns the precision policy
        """
        return self.precision_policy

    def get_grid(self):
        """
        Returns the grid object
        """
        return self.grid

    def _construct_warp(self):
        """
        Construct the warp functional and kernel of the operator
        TODO: Maybe a better way to do this?
        Maybe add this to the compute backend decorator?
        Leave it for now, as it is not clear how the warp compute backend will evolve
        """
        return None, None

    def _construct_neon(self):
        """
        Construct the Neon functional and Neon container of the operator
        TODO: Maybe a better way to do this?
        Maybe add this to the backend decorator?
        Leave it for now, as it is not clear how the neon backend will evolve
        """
        return None, None

    def _construct_read_write_functions(self):
        """Build backend-specific ``read_field`` / ``write_field`` helpers.

        For the Warp backend these are direct 4-D array accesses.  For the
        Neon backend they wrap ``wp.neon_read`` / ``wp.neon_write``.

        Returns
        -------
        tuple of wp.func
            ``(read_field, write_field)``
        """
        if self.compute_backend == ComputeBackend.WARP:

            @wp.func
            def read_field(
                field: Any,
                index: Any,
                direction: Any,
            ):
                # This function reads a field value at a given index and direction.
                return field[direction, index[0], index[1], index[2]]

            @wp.func
            def write_field(
                field: Any,
                index: Any,
                direction: Any,
                value: Any,
            ):
                # This function writes a value to a field at a given index and direction.
                field[direction, index[0], index[1], index[2]] = value

        elif self.compute_backend == ComputeBackend.NEON:
            import neon

            @wp.func
            def read_field(
                field: Any,
                index: Any,
                direction: Any,
            ):
                # This function reads a field value at a given index and direction.
                return wp.neon_read(field, index, direction)

            @wp.func
            def write_field(
                field: Any,
                index: Any,
                direction: Any,
                value: Any,
            ):
                # This function writes a value to a field at a given index and direction.
                wp.neon_write(field, index, direction, value)

        else:
            raise ValueError(f"Unsupported compute backend: {self.compute_backend}")

        return read_field, write_field

    def _construct_read_field_neighbor(self):
        """
        Construct a function to read a field value at a neighboring index along a given direction.
        """

        if self.compute_backend == ComputeBackend.WARP:

            @wp.func
            def read_field_neighbor(
                field: Any,
                index: Any,
                offset: Any,
                direction: Any,
            ):
                # This function reads a field value at a given neighboring index and direction.
                neighbor = index + offset
                return field[direction, neighbor[0], neighbor[1], neighbor[2]]

        elif self.compute_backend == ComputeBackend.NEON:
            import neon
            # from neon.multires.mPartition import neon_get_type

            @wp.func
            def read_field_neighbor(
                field: Any,
                index: Any,
                offset: Any,
                direction: Any,
            ):
                # This function reads a field value at a given neighboring index and direction.
                unused_is_valid = wp.bool(False)
                # dtype = neon_get_type(field)  # This is a placeholder to ensure the dtype is set correctly
                return wp.neon_read_ngh(field, index, offset, direction, wp.uint8(0.0), unused_is_valid)

        else:
            raise ValueError(f"Unsupported compute backend: {self.compute_backend}")

        return read_field_neighbor

import inspect
import traceback
import jax

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
        # Set the default values from the global config
        self.velocity_set = velocity_set or DefaultConfig.velocity_set
        self.precision_policy = precision_policy or DefaultConfig.default_precision_policy
        self.compute_backend = compute_backend or DefaultConfig.default_backend

        # Check if the compute compute_backend is supported
        if self.compute_backend not in ComputeBackend:
            raise ValueError(f"Compute_backend {compute_backend} is not supported")

        # Construct the kernel based compute_backend functions TODO: Maybe move this to the register or something
        if self.compute_backend == ComputeBackend.WARP:
            self.warp_functional, self.warp_kernel = self._construct_warp()

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
            subclass_name = func.__qualname__.split(".")[0]
            signature = inspect.signature(func)
            key = (subclass_name, backend_name, str(signature))
            cls._backends[key] = func
            return func

        return decorator

    def __call__(self, *args, callback=None, **kwargs):
        method_candidates = [
            (key, method) for key, method in self._backends.items() if key[0] == self.__class__.__name__ and key[1] == self.compute_backend
        ]
        bound_arguments = None
        for key, backend_method in method_candidates:
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

        raise Exception(f"Error captured for backend with key {key} for operator {self.__class__.__name__}: {error}\n {traceback_str}")

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

    @property
    def store_dtype(self):
        """
        Returns the store dtype
        """
        if self.compute_backend == ComputeBackend.JAX:
            return self.precision_policy.store_precision.jax_dtype
        elif self.compute_backend == ComputeBackend.WARP:
            return self.precision_policy.store_precision.wp_dtype

    def _construct_warp(self):
        """
        Construct the warp functional and kernel of the operator
        TODO: Maybe a better way to do this?
        Maybe add this to the compute backend decorator?
        Leave it for now, as it is not clear how the warp compute backend will evolve
        """
        return None, None

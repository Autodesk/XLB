# Base class for all operators, (collision, streaming, equilibrium, etc.)
import warp as wp

from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy, Precision
from xlb.global_config import GlobalConfig


class Operator:
    """
    Base class for all operators, collision, streaming, equilibrium, etc.

    This class is responsible for handling compute backends.
    """

    _backends = {}

    def __init__(self, velocity_set, precision_policy, compute_backend):
        # Set the default values from the global config
        self.velocity_set = velocity_set or GlobalConfig.velocity_set
        self.precision_policy = precision_policy or GlobalConfig.precision_policy
        self.compute_backend = compute_backend or GlobalConfig.compute_backend

        # Check if the compute backend is supported
        if self.compute_backend not in ComputeBackend:
            raise ValueError(f"Compute backend {compute_backend} is not supported")

        # Construct the kernel based backend functions TODO: Maybe move this to the register or something
        if self.compute_backend == ComputeBackend.WARP:
            self._functional, self._kernel = self._construct_warp()

    @classmethod
    def register_backend(cls, backend_name):
        """
        Decorator to register a backend for the operator.
        """

        def decorator(func):
            # Use the combination of operator name and backend name as the key
            subclass_name = func.__qualname__.split(".")[0]
            key = (subclass_name, backend_name)
            cls._backends[key] = func
            return func

        return decorator

    def __call__(self, *args, callback=None, **kwargs):
        """
        Calls the operator with the compute backend specified in the constructor.
        If a callback is provided, it is called either with the result of the operation
        or with the original arguments and keyword arguments if the backend modifies them by reference.
        """
        key = (self.__class__.__name__, self.compute_backend)
        backend_method = self._backends.get(key)

        if backend_method:
            result = backend_method(self, *args, **kwargs)

            # Determine what to pass to the callback based on the backend behavior
            callback_arg = result if result is not None else (args, kwargs)

            if callback and callable(callback):
                callback(callback_arg)

            return result
        else:
            raise NotImplementedError(f"Backend {self.compute_backend} not implemented")

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
            import jax as backend
        elif self.compute_backend == ComputeBackend.WARP:
            import warp as backend
        return backend

    @property
    def compute_dtype(self):
        """
        Returns the compute dtype
        """
        return self._precision_to_dtype(self.precision_policy.compute_precision)

    @property
    def store_dtype(self):
        """
        Returns the store dtype
        """
        return self._precision_to_dtype(self.precision_policy.store_precision)

    @staticmethod
    def _precision_to_dtype(precision):
        """
        Convert the precision to the corresponding dtype
        TODO: Maybe move this
        """
        if precision == Precision.FP64:
            return self.backend.float64
        elif precision == Precision.FP32:
            return self.backend.float32
        elif precision == Precision.FP16:
            return self.backend.float16

    ### WARP specific types ###
    # These are used to define the types for the warp backend
    # TODO: There might be a better place to put these
    @property
    def _warp_u_vec(self):
        """
        Returns the warp type for velocity
        """
        return wp.vec(self.velocity_set.d, dtype=self.compute_dtype)

    @property
    def _warp_lattice_vec(self):
        """
        Returns the warp type for the lattice
        """
        return wp.vec(self.velocity_set.q, dtype=self.compute_dtype)

    @property
    def _warp_int_lattice_vec(self):
        """
        Returns the warp type for the streaming matrix (c)
        """
        return wp.vec(self.velocity_set.q, dtype=wp.int32)

    @property
    def _warp_bool_lattice_vec(self):
        """
        Returns the warp type for the streaming matrix (c)
        """
        return wp.vec(self.velocity_set.q, dtype=wp.bool)

    @property
    def _warp_stream_mat(self):
        """
        Returns the warp type for the streaming matrix (c)
        """
        return wp.mat(
            (self.velocity_set.d, self.velocity_set.q), dtype=self.compute_dtype
        )

    @property
    def _warp_array_type(self):
        """
        Returns the warp type for arrays
        """
        if self.velocity_set.d == 2:
            return wp.array3d(dtype=self.store_dtype)
        elif self.velocity_set.d == 3:
            return wp.array4d(dtype=self.store_dtype)

    @property
    def _warp_uint8_array_type(self):
        """
        Returns the warp type for arrays
        """
        if self.velocity_set.d == 2:
            return wp.array3d(dtype=wp.bool)
        elif self.velocity_set.d == 3:
            return wp.array4d(dtype=wp.bool)

    @property
    def _warp_bool_array_type(self):
        """
        Returns the warp type for arrays
        """
        if self.velocity_set.d == 2:
            return wp.array3d(dtype=wp.bool)
        elif self.velocity_set.d == 3:
            return wp.array4d(dtype=wp.bool)

    def _construct_warp(self):
        """
        Construct the warp functional and kernel of the operator
        TODO: Maybe a better way to do this?
        Maybe add this to the backend decorator?
        """
        raise NotImplementedError("Children must implement this method")

# Base class for all operators, (collision, streaming, equilibrium, etc.)

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend


class Operator:
    """
    Base class for all operators, collision, streaming, equilibrium, etc.

    This class is responsible for handling compute backends.
    """

    def __init__(self, velocity_set, compute_backend):
        self.velocity_set = velocity_set
        self.compute_backend = compute_backend

        # Check if compute backend is supported
        # TODO: Fix check for compute backend
        #if self.compute_backend not in self.supported_compute_backends:
        #    raise ValueError(
        #        f"Compute backend {self.compute_backend} not supported by {self.__class__.__name__}"
        #    )

    def __call__(self, *args, **kwargs):
        """
        Apply the operator to a input. This method will call the
        appropriate apply method based on the compute backend.
        """
        if self.compute_backend == ComputeBackend.JAX:
            return self.apply_jax(*args, **kwargs)
        elif self.compute_backend == ComputeBackend.NUMBA:
            return self.apply_numba(*args, **kwargs)

    def apply_jax(self, *args, **kwargs):
        """
        Implement the operator using JAX.
        If using the JAX backend, this method will then become
        the self.__call__ method.
        """
        raise NotImplementedError("Child class must implement apply_jax")

    def apply_numba(self, *args, **kwargs):
        """
        Implement the operator using Numba.
        If using the Numba backend, this method will then become
        the self.__call__ method.
        """
        raise NotImplementedError("Child class must implement apply_numba")

    def construct_numba(self):
        """
        Constructs numba kernel for the operator
        """
        raise NotImplementedError("Child class must implement apply_numba")

    @property
    def supported_compute_backend(self):
        """
        Returns the supported compute backend for the operator
        """
        supported_backend = []
        if self._is_method_overridden("apply_jax"):
            supported_backend.append(ComputeBackend.JAX)
        elif self._is_method_overridden("apply_numba"):
            supported_backend.append(ComputeBackend.NUMBA)
        else:
            raise NotImplementedError("No supported compute backend implemented")
        return supported_backend

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

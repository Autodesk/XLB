# Base class for all operators, (collision, streaming, equilibrium, etc.)

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backends import ComputeBackends


class Operator:
    """
    Base class for all operators, collision, streaming, equilibrium, etc.

    This class is responsible for handling compute backends.
    """

    _backends = {}

    def __init__(self, velocity_set, compute_backend):
        self.velocity_set = velocity_set
        self.compute_backend = compute_backend
        if compute_backend not in ComputeBackends:
            raise ValueError(f"Compute backend {compute_backend} is not supported")

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

    def __call__(self, *args, **kwargs):
        """
        Calls the operator with the compute backend specified in the constructor.
        """
        key = (self.__class__.__name__, self.compute_backend)
        backend_method = self._backends.get(key)
        if backend_method:
            return backend_method(self, *args, **kwargs)
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

    def data_handler(self, *args, **kwargs):
        """
        Handles data for the operator.
        """
        raise NotImplementedError("Child class must implement data_handler")

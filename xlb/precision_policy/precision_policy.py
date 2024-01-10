class PrecisionPolicy(object):
    """
    Base class for precision policy in lattice Boltzmann method.
    Stores dtype information and provides an interface for casting operations.
    """
    def __init__(self, compute_dtype, storage_dtype):
        self.compute_dtype = compute_dtype
        self.storage_dtype = storage_dtype

    def cast_to_compute(self, array):
        """
        Cast the array to the computation precision.
        To be implemented by subclass.
        """
        raise NotImplementedError

    def cast_to_store(self, array):
        """
        Cast the array to the storage precision.
        To be implemented by subclass.
        """
        raise NotImplementedError
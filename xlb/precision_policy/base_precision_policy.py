from abc import ABC, abstractmethod

class PrecisionPolicy(ABC):
    def __init__(self, compute_dtype, storage_dtype):
        self.compute_dtype = compute_dtype
        self.storage_dtype = storage_dtype

    @abstractmethod
    def cast_to_compute(self, array):
        pass

    @abstractmethod
    def cast_to_store(self, array):
        pass

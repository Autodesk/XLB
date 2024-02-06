# Enum for precision policy

from enum import Enum, auto

class Precision(Enum):
    FP64 = auto()
    FP32 = auto()
    FP16 = auto()

class PrecisionPolicy(Enum):
    FP64FP64 = auto()
    FP64FP32 = auto()
    FP64FP16 = auto()
    FP32FP32 = auto()
    FP32FP16 = auto()

    @property
    def compute_precision(self):
        if self == PrecisionPolicy.FP64FP64:
            return Precision.FP64
        elif self == PrecisionPolicy.FP64FP32:
            return Precision.FP32
        elif self == PrecisionPolicy.FP64FP16:
            return Precision.FP16
        elif self == PrecisionPolicy.FP32FP32:
            return Precision.FP32
        elif self == PrecisionPolicy.FP32FP16:
            return Precision.FP16
        else:
            raise ValueError("Invalid precision policy")

    @property
    def store_precision(self):
        if self == PrecisionPolicy.FP64FP64:
            return Precision.FP64
        elif self == PrecisionPolicy.FP64FP32:
            return Precision.FP32
        elif self == PrecisionPolicy.FP64FP16:
            return Precision.FP16
        elif self == PrecisionPolicy.FP32FP32:
            return Precision.FP32
        elif self == PrecisionPolicy.FP32FP16:
            return Precision.FP16
        else:
            raise ValueError("Invalid precision policy")

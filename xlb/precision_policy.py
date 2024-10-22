# Enum for precision policy

from enum import Enum, auto
import jax.numpy as jnp
import warp as wp


class Precision(Enum):
    FP64 = auto()
    FP32 = auto()
    FP16 = auto()
    UINT8 = auto()
    BOOL = auto()

    @property
    def wp_dtype(self):
        if self == Precision.FP64:
            return wp.float64
        elif self == Precision.FP32:
            return wp.float32
        elif self == Precision.FP16:
            return wp.float16
        elif self == Precision.UINT8:
            return wp.uint8
        elif self == Precision.BOOL:
            return wp.bool
        else:
            raise ValueError("Invalid precision")

    @property
    def jax_dtype(self):
        if self == Precision.FP64:
            return jnp.float64
        elif self == Precision.FP32:
            return jnp.float32
        elif self == Precision.FP16:
            return jnp.float16
        elif self == Precision.UINT8:
            return jnp.uint8
        elif self == Precision.BOOL:
            return jnp.bool_
        else:
            raise ValueError("Invalid precision")


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
            return Precision.FP64
        elif self == PrecisionPolicy.FP64FP16:
            return Precision.FP64
        elif self == PrecisionPolicy.FP32FP32:
            return Precision.FP32
        elif self == PrecisionPolicy.FP32FP16:
            return Precision.FP32
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

    def cast_to_compute_jax(self, array):
        compute_precision = self.compute_precision
        return jnp.array(array, dtype=compute_precision.jax_dtype)

    def cast_to_store_jax(self, array):
        store_precision = self.store_precision
        return jnp.array(array, dtype=store_precision.jax_dtype)

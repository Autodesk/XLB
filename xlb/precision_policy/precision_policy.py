from abc import ABC, abstractmethod
from xlb.compute_backend import ComputeBackend
from xlb.global_config import GlobalConfig

from xlb.precision_policy.jax_precision_policy import (
    JaxFp32Fp32,
    JaxFp32Fp16,
    JaxFp64Fp64,
    JaxFp64Fp32,
    JaxFp64Fp16,
)


class Fp64Fp64:
    def __new__(cls):
        if (
            GlobalConfig.compute_backend == ComputeBackend.JAX
            or GlobalConfig.compute_backend == ComputeBackend.PALLAS
        ):
            return JaxFp64Fp64()
        else:
            raise ValueError(
                f"Unsupported compute backend: {GlobalConfig.compute_backend}"
            )


class Fp64Fp32:
    def __new__(cls):
        if (
            GlobalConfig.compute_backend == ComputeBackend.JAX
            or GlobalConfig.compute_backend == ComputeBackend.PALLAS
        ):
            return JaxFp64Fp32()
        else:
            raise ValueError(
                f"Unsupported compute backend: {GlobalConfig.compute_backend}"
            )


class Fp32Fp32:
    def __new__(cls):
        if (
            GlobalConfig.compute_backend == ComputeBackend.JAX
            or GlobalConfig.compute_backend == ComputeBackend.PALLAS
        ):
            return JaxFp32Fp32()
        else:
            raise ValueError(
                f"Unsupported compute backend: {GlobalConfig.compute_backend}"
            )


class Fp64Fp16:
    def __new__(cls):
        if (
            GlobalConfig.compute_backend == ComputeBackend.JAX
            or GlobalConfig.compute_backend == ComputeBackend.PALLAS
        ):
            return JaxFp64Fp16()
        else:
            raise ValueError(
                f"Unsupported compute backend: {GlobalConfig.compute_backend}"
            )


class Fp32Fp16:
    def __new__(cls):
        if (
            GlobalConfig.compute_backend == ComputeBackend.JAX
            or GlobalConfig.compute_backend == ComputeBackend.PALLAS
        ):
            return JaxFp32Fp16()
        else:
            raise ValueError(
                f"Unsupported compute backend: {GlobalConfig.compute_backend}"
            )

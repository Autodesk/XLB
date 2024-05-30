from xlb.compute_backend import ComputeBackend
from dataclasses import dataclass


@dataclass
class DefaultConfig:
    default_precision_policy = None
    velocity_set = None
    default_backend = None


def init(velocity_set, default_backend, default_precision_policy):
    DefaultConfig.velocity_set = velocity_set
    DefaultConfig.default_backend = default_backend
    DefaultConfig.default_precision_policy = default_precision_policy

    if default_backend == ComputeBackend.WARP:
        import warp as wp

        wp.init()
    elif default_backend == ComputeBackend.JAX:
        check_multi_gpu_support()
    else:
        raise ValueError(f"Unsupported compute backend: {default_backend}")


def default_backend() -> ComputeBackend:
    return DefaultConfig.default_backend


def check_multi_gpu_support():
    import jax

    gpus = jax.devices("gpu")
    if len(gpus) > 1:
        print("Multi-GPU support is available: {} GPUs detected.".format(len(gpus)))
    elif len(gpus) == 1:
        print("Single-GPU support is available: 1 GPU detected.")
    else:
        print("No GPU support is available; CPU fallback will be used.")

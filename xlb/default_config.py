"""
Global configuration for XLB.

Call :func:`init` once at the start of every script to select the velocity
set, compute backend, and precision policy.  All operators read their
defaults from :class:`DefaultConfig` when explicit arguments are omitted.
"""

import contextlib
import os

from xlb.compute_backend import ComputeBackend
from dataclasses import dataclass
from xlb.precision_policy import PrecisionPolicy


@dataclass
class DefaultConfig:
    """Singleton holding the active global configuration.

    Attributes are set by :func:`init` and read by operators, grids, and
    helpers throughout XLB.

    Attributes
    ----------
    default_precision_policy : PrecisionPolicy or None
        Active precision policy (compute / store dtype pair).
    velocity_set : VelocitySet or None
        Active lattice velocity set.
    default_backend : ComputeBackend or None
        Active compute backend.
    enable_backward : bool or None
        Whether Warp generates adjoint (backward) kernels; resolved by
        :func:`init`. ``None`` until initialized, and stays ``None`` on the JAX
        backend, which does not use Warp.
    """

    default_precision_policy = None
    velocity_set = None
    default_backend = None
    enable_backward = None


_ENABLE_BACKWARD_ENV = "XLB_WARP_ENABLE_BACKWARD"
_TRUTHY = {"1", "true", "yes", "on"}


def _resolve_enable_backward(enable_backward):
    """Decide whether Warp should generate adjoint (backward) kernels.

    An explicit ``True``/``False`` from :func:`init` wins. When it is ``None``,
    fall back to ``XLB_WARP_ENABLE_BACKWARD`` and then to off.
    """
    if enable_backward is not None:
        return bool(enable_backward)
    return os.environ.get(_ENABLE_BACKWARD_ENV, "").strip().lower() in _TRUTHY


def _configure_warp_backward_codegen(enable_backward):
    """Apply the resolved backward-codegen choice to Warp's global config.

    Warp emits an adjoint version of every kernel by default, which roughly
    doubles codegen and compile time. XLB's LBM solvers are forward-only, and on
    the Neon backend the legacy ``@wp.func`` patterns fail NVRTC adjoint
    compilation outright, so XLB leaves adjoints off unless asked.
    """
    import warp as wp

    wp.config.enable_backward = enable_backward


def _warp_init_and_select_cuda_device(enable_backward):
    """Initialize Warp and pin the default CUDA device for single-GPU XLB runs.

    With multiple GPUs, Warp's default device for allocations and launches can
    otherwise diverge. Set ``XLB_WARP_DEVICE`` (e.g. ``cuda:0`` or ``cuda:1``)
    to choose which GPU Warp uses; defaults to ``cuda:0`` when unset.
    """
    import warp as wp

    # Must precede kernel construction: Warp captures enable_backward into a
    # module's options when that module's first kernel is created.
    _configure_warp_backward_codegen(enable_backward)

    wp.init()  # TODO: Must be removed in the future versions of WARP
    if wp.get_cuda_device_count() == 0:
        return
    choice = os.environ.get("XLB_WARP_DEVICE", "cuda:0").strip()
    try:
        wp.set_device(choice)
    except Exception:
        # Fall back to the first GPU; if that fails too, leave Warp's own default.
        with contextlib.suppress(Exception):
            wp.set_device("cuda:0")


def init(velocity_set, default_backend, default_precision_policy, enable_backward=None):
    """Initialize the global XLB configuration.

    Must be called before creating any grid, operator, or field.

    Parameters
    ----------
    velocity_set : VelocitySet
        Lattice velocity set (e.g. ``D3Q19``).
    default_backend : ComputeBackend
        Compute backend to use (JAX, WARP, or NEON).
    default_precision_policy : PrecisionPolicy
        Precision policy for compute and storage dtypes.
    enable_backward : bool, optional
        Whether Warp generates adjoint (backward) kernels. Pass ``True`` when
        differentiating through the solver with ``wp.Tape``; leaving it off
        roughly halves kernel codegen and compile time, and is required on the
        Neon backend, whose ``@wp.func`` patterns fail NVRTC adjoint
        compilation. Defaults to the ``XLB_WARP_ENABLE_BACKWARD`` environment
        variable, or off when that is unset. Ignored on the JAX backend.
    """
    DefaultConfig.velocity_set = velocity_set
    DefaultConfig.default_backend = default_backend
    DefaultConfig.default_precision_policy = default_precision_policy

    if default_backend == ComputeBackend.WARP:
        DefaultConfig.enable_backward = _resolve_enable_backward(enable_backward)
        _warp_init_and_select_cuda_device(DefaultConfig.enable_backward)
    elif default_backend == ComputeBackend.NEON:
        import warp as wp
        import neon

        # wp.config.mode = "release"
        # wp.config.llvm_cuda = False
        # wp.config.verbose = True
        # wp.verbose_warnings = True

        DefaultConfig.enable_backward = _resolve_enable_backward(enable_backward)
        _warp_init_and_select_cuda_device(DefaultConfig.enable_backward)

        # It's a good idea to always clear the kernel cache when developing new native or codegen features
        wp.clear_kernel_cache()

        # !!! DO THIS BEFORE DEFINING/USING ANY KERNELS WITH CUSTOM TYPES
        neon.init()

    elif default_backend == ComputeBackend.JAX:
        check_backend_support()
    else:
        raise ValueError(f"Unsupported compute backend: {default_backend}")


def default_backend() -> ComputeBackend:
    """Return the currently configured compute backend."""
    return DefaultConfig.default_backend


def check_backend_support():
    """Print a summary of available JAX hardware accelerators."""
    import jax

    if jax.devices()[0].platform == "gpu":
        gpus = jax.devices("gpu")
        if len(gpus) > 1:
            print("Multi-GPU support is available: {} GPUs detected.".format(len(gpus)))
        elif len(gpus) == 1:
            print("Single-GPU support is available: 1 GPU detected.")

    elif jax.devices()[0].platform == "tpu":
        tpus = jax.devices("tpu")
        if len(tpus) > 1:
            print("Multi-TPU support is available: {} TPUs detected.".format(len(tpus)))
        elif len(tpus) == 1:
            print("Single-TPU support is available: 1 TPU detected.")
    else:
        print("No GPU support is available; CPU fallback will be used.")

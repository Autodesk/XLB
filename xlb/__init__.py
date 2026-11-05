import warnings
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("xlb")
except PackageNotFoundError:
    __version__ = "0.0.0"


def _installed(distribution):
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _check_warp_install():
    """Report the two ways a Warp install can be wrong before the failure gets cryptic.

    Warp comes from either the ``[warp]`` extra (``warp-lang``) or the ``[neon]``
    extra (``neon_gpu``, which bundles a fork). Both unpack into
    ``site-packages/warp``, so having both means whichever pip wrote last owns
    the directory and uninstalling ``warp-lang`` would delete files Neon needs.
    XLB no longer pulls ``warp-lang`` in as a core dependency, but pip does not
    remove one that was already there, so warn rather than assume.
    """
    try:
        import warp  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "warp":
            raise
        raise ModuleNotFoundError(
            "XLB requires Warp, which is not installed. Install exactly one of "
            "'xlb[warp]' (Warp from PyPI) or 'xlb[neon]' (Neon's bundled Warp fork)."
        ) from exc

    warp_lang, neon_gpu = _installed("warp-lang"), _installed("neon_gpu")
    if warp_lang and neon_gpu:
        warnings.warn(
            f"Both warp-lang {warp_lang} and neon_gpu {neon_gpu} are installed, and they share "
            f"site-packages/warp; warp {warp.__version__} currently owns it. Run "
            "'pip uninstall warp-lang' and reinstall 'xlb[neon]' to restore Neon's fork, "
            "or drop neon_gpu if you meant to use 'xlb[warp]'.",
            RuntimeWarning,
            stacklevel=2,
        )


_check_warp_install()

# Enum classes
from xlb.compute_backend import ComputeBackend as ComputeBackend
from xlb.precision_policy import PrecisionPolicy as PrecisionPolicy, Precision as Precision
from xlb.physics_type import PhysicsType as PhysicsType
from xlb.mres_perf_optimization_type import MresPerfOptimizationType as MresPerfOptimizationType

# Config
from .default_config import init as init, DefaultConfig as DefaultConfig

# Velocity Set
import xlb.velocity_set

# Operators
import xlb.operator.equilibrium
import xlb.operator.collision
import xlb.operator.stream
import xlb.operator.boundary_condition
import xlb.operator.macroscopic
import xlb.operator.postprocess

# Grids
import xlb.grid

# Solvers
import xlb.helper

# Utils
import xlb.utils

# Distributed computing
import xlb.distribute

import platform
import sys

from setuptools import setup, find_packages

# Warp is required at import time, but it is deliberately not in
# ``install_requires``: it must come from exactly one of the two extras.
# ``[warp]`` installs ``warp-lang`` from PyPI, while ``[neon]`` installs
# ``neon_gpu``, which bundles its own Warp fork. Both unpack into
# ``site-packages/warp``, so if pip installs both, whichever lands last owns the
# directory and later uninstalling ``warp-lang`` deletes files Neon needs.
#
# Listing ``warp-lang`` as a core dependency made that collision unavoidable for
# ``pip install xlb[neon]``. A setup.py hook cannot undo it either: pip installs
# dependencies after building this project, and for wheel and PEP 660 editable
# installs the ``install`` command never runs at all. Keeping it out of the core
# dependencies is the only way ``[neon]`` reliably gets one Warp.
_NEON_VERSION = "0.5.2a3"
_NEON_RELEASE_URL = f"https://github.com/Autodesk/Neon/releases/download/v{_NEON_VERSION}"


def _neon_wheel_requirement():
    """Build a direct-reference requirement for the neon_gpu wheel matching the running Python."""
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    machine = platform.machine()
    plat = "linux_aarch64" if machine == "aarch64" else "linux_x86_64"
    wheel = f"neon_gpu-{_NEON_VERSION}-{tag}-{tag}-{plat}.whl"
    url = f"{_NEON_RELEASE_URL}/{wheel}"
    req = f"neon_gpu @ {url}"
    print(f"[xlb] Neon wheel for Python {sys.version_info.major}.{sys.version_info.minor} ({plat}): {url}")
    print(f"[xlb] Neon requirement: {req}")
    return req


setup(
    name="xlb",
    version="0.3.2",
    description="XLB: Accelerated Lattice Boltzmann (XLB) for Physics-based ML",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mehdi Ataei",
    url="https://github.com/Autodesk/XLB",
    license="Apache License 2.0",
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.9.2",
        "numpy>=2.1.2",
        "pyvista>=0.44.1",
        "trimesh>=4.4.9",
        "numpy-stl>=3.1.2",
        "pydantic>=2.9.1",
        "ruff>=0.14.1",
        "jax>=0.8.2",  # Base JAX CPU-only requirement
        "nvtx>=0.2.0",  # NVTX ranges (e.g. nse_multires_stepper); listed in requirements.txt
    ],
    extras_require={
        # Exactly one of [warp] and [neon] is required; see the note above.
        # Bounded to the versions the suite has been run against: 1.16, the API
        # XLB's kernels target and the series of the fork neon_gpu bundles, and
        # 1.17. A fresh install takes 1.17; pin warp-lang==1.16.* to match Neon.
        "warp": ["warp-lang>=1.16,<1.18"],
        "cuda": ["jax[cuda13]>=0.8.2"],  # For CUDA installations (pip install -U "jax[cuda13]")
        "tpu": ["jax[tpu]>=0.8.2"],  # For TPU installations
        # h5py: MultiresIO / Neon multi-resolution export to HDF5 (see xlb.utils.mesher).
        "neon": [_neon_wheel_requirement(), "h5py>=3.10.0"],
        "test": ["pytest>=8.0.0"],
    },
    python_requires=">=3.11",
    dependency_links=["https://storage.googleapis.com/jax-releases/libtpu_releases.html"],
)

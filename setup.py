import os
import platform
import subprocess
import sys

from setuptools import setup, find_packages
from setuptools.command.install import install


def _neon_extra_requested():
    """Best-effort detection of [neon] extra from install invocation."""
    for arg in sys.argv:
        if "neon" in arg and ("[" in arg or "xlb" in arg):
            return True
    return False


def _uninstall_warp_lang(*, reason: str) -> None:
    """Uninstall the ``warp-lang`` distribution so Neon's bundled warp fork is used."""
    if os.environ.get("XLB_NEON_SKIP_UNINSTALL_WARP", "").lower() in ("1", "true", "yes"):
        return
    print(f"[xlb] {reason}")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "warp-lang", "-y"],
            check=False,
            capture_output=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[xlb] Warning: failed to uninstall warp-lang: {exc}")


_NEON_VERSION = "0.5.2a1"
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


class InstallWithNeonHooks(install):
    """Uninstall ``warp-lang`` before and after install when ``[neon]`` is requested.

    * **Before** ``pip``/setuptools install dependencies: removes any previously
      installed ``warp-lang`` so an older or PyPI build does not linger next to
      Neon's fork (``neon_gpu`` ships its own warp).
    * **After** install: removes the ``warp-lang`` pulled in by ``install_requires``,
      leaving Neon's warp as the one on the path.

    Only runs when installing from source (e.g. sdist or git). Wheel installs
    do not run setup.py, so for ``pip install xlb[neon]`` from PyPI you may
    need to run ``pip uninstall warp-lang`` first if it is already installed.
    Set XLB_NEON_SKIP_UNINSTALL_WARP=1 to disable this behaviour.
    """

    def run(self):
        if _neon_extra_requested():
            _uninstall_warp_lang(
                reason=("Removing any existing warp-lang before Neon install (neon_gpu provides its own warp fork)."),
            )
        install.run(self)
        if _neon_extra_requested():
            _uninstall_warp_lang(
                reason=("Removing PyPI warp-lang after install (core deps); use the warp bundled with neon_gpu."),
            )


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
        "warp-lang>=1.10.0",  # Required at import time (core modules import warp)
        "nvtx>=0.2.0",  # NVTX ranges (e.g. nse_multires_stepper); listed in requirements.txt
    ],
    extras_require={
        "warp": ["warp-lang>=1.10.0"],  # Kept for explicit `pip install xlb[warp]` / Neon uninstall hook docs
        "cuda": ["jax[cuda13]>=0.8.2"],  # For CUDA installations (pip install -U "jax[cuda13]")
        "tpu": ["jax[tpu]>=0.8.2"],  # For TPU installations
        # h5py: MultiresIO / Neon multi-resolution export to HDF5 (see xlb.utils.mesher).
        "neon": [_neon_wheel_requirement(), "h5py>=3.10.0"],
        "test": ["pytest>=8.0.0"],
    },
    python_requires=">=3.11",
    dependency_links=["https://storage.googleapis.com/jax-releases/libtpu_releases.html"],
    cmdclass={"install": InstallWithNeonHooks},
)

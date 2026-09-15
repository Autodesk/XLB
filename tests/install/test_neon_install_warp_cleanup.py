"""
Verify that ``pip install -e .[neon]`` leaves exactly one Warp on the path.

``neon_gpu`` bundles a Warp fork at the same ``site-packages/warp`` as PyPI's
``warp-lang``, so if both are installed whichever pip unpacked last owns the
directory. XLB therefore keeps ``warp-lang`` out of ``install_requires`` and
offers it only through the ``[warp]`` extra; these tests pin that down, since a
mixed install still imports and passes the rest of the suite.

Uses an isolated virtualenv and ``pip install -e``, same style as
``run_install_matrix.py``. Skips on platforms where Neon wheels are not built
(see ``setup.py``).
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

requires_neon_wheel = pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"),
    reason="Neon wheels: Linux x86_64 / aarch64 only",
)
requires_py311 = pytest.mark.skipif(sys.version_info < (3, 11), reason="XLB requires Python >= 3.11")


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def _make_venv(venv_dir: Path) -> Path:
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    venv_py = _venv_python(venv_dir)
    assert venv_py.is_file(), f"missing venv python: {venv_py}"
    proc = _run([str(venv_py), "-m", "pip", "install", "--upgrade", "pip", "wheel"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return venv_py


def _pip_install(venv_py: Path, *args: str) -> None:
    proc = _run([str(venv_py), "-m", "pip", "install", *args])
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _distribution_version(venv_py: Path, distribution: str) -> str | None:
    """Return the installed version of *distribution*, or None if absent."""
    code = (
        "from importlib.metadata import PackageNotFoundError, version\n"
        "try:\n"
        f"    print(version({distribution!r}))\n"
        "except PackageNotFoundError:\n"
        "    print('')\n"
    )
    proc = _run([str(venv_py), "-c", code])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout.strip() or None


@requires_py311
@requires_neon_wheel
def test_neon_install_does_not_pull_warp_lang(tmp_path: Path) -> None:
    """A clean ``[neon]`` install must get its Warp from Neon and nowhere else."""
    venv_py = _make_venv(tmp_path / "venv")
    _pip_install(venv_py, "-e", ".[neon,test]")

    assert _distribution_version(venv_py, "neon_gpu") is not None, "neon_gpu was not installed"
    warp_lang = _distribution_version(venv_py, "warp-lang")
    assert warp_lang is None, (
        f"warp-lang {warp_lang} came in alongside neon_gpu; both own site-packages/warp, "
        "so whichever pip unpacked last wins. Keep warp-lang out of install_requires."
    )

    proc = _run([str(venv_py), "-c", "import neon, warp; print(warp.__version__, warp.__file__)"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "dev" in proc.stdout, f"expected Neon's Warp fork, got: {proc.stdout.strip()}"

    # A repeated editable install must not disturb the result.
    _pip_install(venv_py, "-e", ".[neon,test]")
    assert _distribution_version(venv_py, "warp-lang") is None, "warp-lang reappeared on reinstall"


@requires_py311
@requires_neon_wheel
def test_preexisting_warp_lang_is_reported(tmp_path: Path) -> None:
    """pip will not remove a warp-lang that predates the install, so XLB must warn."""
    venv_py = _make_venv(tmp_path / "venv")
    _pip_install(venv_py, "warp-lang==1.10.0")
    _pip_install(venv_py, "-e", ".[neon,test]")

    code = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import xlb\n"
        "print([str(w.message) for w in caught])\n"
    )
    proc = _run([str(venv_py), "-c", code])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "warp-lang" in proc.stdout and "neon_gpu" in proc.stdout, (
        f"importing XLB with both warp-lang and neon_gpu should warn, got: {proc.stdout.strip()}"
    )

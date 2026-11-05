"""
Verify that installing XLB with the ``[neon]`` extra uninstalls a pre-existing
``warp-lang`` (Neon ships its own fork via ``neon_gpu``).

Uses an isolated virtualenv and ``pip install -e``, same style as
``run_install_matrix.py``. Skips on platforms where Neon wheels are not built
(see ``setup.py``).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _neon_wheels_supported() -> bool:
    if sys.platform != "linux":
        return False
    return platform.machine() in ("x86_64", "aarch64")


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.skipif(sys.version_info < (3, 11), reason="XLB requires Python >= 3.11")
@pytest.mark.skipif(not _neon_wheels_supported(), reason="Neon wheels: Linux x86_64 / aarch64 only")
def test_neon_editable_install_after_prior_warp_lang(tmp_path: Path) -> None:
    """Pre-install ``warp-lang``, then ``pip install -e .[neon,test]``; imports must work."""
    venv_dir = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    venv_py = _venv_python(venv_dir)
    assert venv_py.is_file(), f"missing venv python: {venv_py}"

    env = {**os.environ, "XLB_NEON_SKIP_UNINSTALL_WARP": ""}

    proc = _run([str(venv_py), "-m", "pip", "install", "--upgrade", "pip", "wheel"], cwd=REPO_ROOT, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    proc = _run(
        [str(venv_py), "-m", "pip", "install", "warp-lang==1.10.0"],
        cwd=REPO_ROOT,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    proc = _run(
        [str(venv_py), "-m", "pip", "install", "-e", ".[neon,test]"],
        cwd=REPO_ROOT,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    proc = _run(
        [str(venv_py), "-c", "import neon; import warp; print('ok', neon.__file__, warp.__file__)"],
        cwd=REPO_ROOT,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok" in proc.stdout

    # Repeat editable install: pre/post uninstall hooks must stay safe on re-run.
    proc = _run(
        [str(venv_py), "-m", "pip", "install", "-e", ".[neon,test]"],
        cwd=REPO_ROOT,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

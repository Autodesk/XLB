#!/usr/bin/env python3
"""
Create isolated virtualenvs, install XLB under several dependency profiles, run
the 3D flow-past-sphere install smoke test, and print a full summary.

Usage (from repository root)::

    python tests/install/run_install_matrix.py

Options::

    --repo-root PATH       Repository root (default: parent of tests/install)
    --reuse-venvs          Do not recreate venvs if they already exist
    --skip-install         Only run tests (assume venvs already populated)

Environment::

    XLB_INSTALL_VENV_ROOT  Override directory for venvs (default: .xlb_install_test_venvs)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


@dataclass
class ScenarioResult:
    name: str
    label: str
    pip_spec: str
    pip_ok: bool = False
    pip_error: str = ""
    sphere_ok: bool | None = None
    sphere_exit: int | None = None
    sphere_error: str = ""


def _run(
    cmd: list[str],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def ensure_venv(venv_dir: Path, *, reuse: bool) -> Path:
    py = _venv_python(venv_dir)
    if reuse and py.is_file():
        return py
    if venv_dir.exists() and not reuse:
        shutil.rmtree(venv_dir)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    return _venv_python(venv_dir)


def pip_install_editable(venv_py: Path, repo_root: Path, extras: str) -> tuple[bool, str]:
    """Run ``pip install -e .[extras]`` from *repo_root*."""
    spec = f".{extras}" if extras else "."
    proc = _run(
        [str(venv_py), "-m", "pip", "install", "--upgrade", "pip", "wheel"],
        cwd=repo_root,
    )
    if proc.returncode != 0:
        return False, proc.stderr + proc.stdout
    proc = _run([str(venv_py), "-m", "pip", "install", "-e", spec], cwd=repo_root)
    if proc.returncode != 0:
        return False, proc.stderr + proc.stdout
    return True, ""


def run_test_script(venv_py: Path, repo_root: Path, test_script: Path) -> tuple[int, str]:
    proc = _run([str(venv_py), str(test_script)], cwd=repo_root)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def main() -> int:
    parser = argparse.ArgumentParser(description="XLB install matrix + 3D sphere smoke test")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="XLB repository root",
    )
    parser.add_argument("--reuse-venvs", action="store_true", help="Reuse existing venv directories")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install; only run tests")
    args = parser.parse_args()

    repo_root: Path = args.repo_root.resolve()
    sphere_script = repo_root / "tests" / "install" / "flow_past_sphere_3d_test.py"
    if not sphere_script.is_file():
        print(f"ERROR: test script not found: {sphere_script}", file=sys.stderr)
        return 2

    venv_root = Path(os.environ.get("XLB_INSTALL_VENV_ROOT", repo_root / ".xlb_install_test_venvs")).resolve()

    scenarios: list[tuple[str, str, str, str]] = [
        (
            "jax-cpu",
            "JAX (CPU): editable install with test extra",
            "[test]",
            "Base JAX CPU jaxlib + test deps (pytest). Warp-lang is still pulled in as a core dependency.",
        ),
        (
            "jax-cuda",
            "JAX[cuda]: editable install with cuda + test extras",
            "[cuda,test]",
            "Adds the cuda extra from setup (jax[cuda13] per setup.py). Requires a matching CUDA stack.",
        ),
        (
            "warp",
            "WARP: explicit [warp,test] extras",
            "[warp,test]",
            "Explicit WARP extra plus test deps; core install already includes warp-lang.",
        ),
        (
            "neon",
            "NEON: editable install with neon + test extras",
            "[neon,test]",
            "Installs neon_gpu wheel per setup.py (Linux x86_64/aarch64, Python 3.11+). "
            "Includes h5py for Neon multires HDF5 export. "
            "Uninstalls any existing warp-lang before install and PyPI warp-lang after (Neon's fork).",
        ),
    ]

    results: list[ScenarioResult] = []

    for folder_name, label, extras, note in scenarios:
        sr = ScenarioResult(name=folder_name, label=label, pip_spec=f"pip install -e .{extras}")
        venv_dir = venv_root / folder_name
        print(f"\n{'=' * 72}\nScenario: {folder_name}\n{label}\nNote: {note}\n{'=' * 72}")

        try:
            venv_py = ensure_venv(venv_dir, reuse=args.reuse_venvs)
        except Exception as exc:
            sr.pip_ok = False
            sr.pip_error = f"venv creation failed: {exc}"
            results.append(sr)
            print(sr.pip_error)
            continue

        if not args.skip_install:
            ok, err = pip_install_editable(venv_py, repo_root, extras)
            sr.pip_ok = ok
            sr.pip_error = err
            if not ok:
                print("PIP INSTALL FAILED\n", err[-4000:] if len(err) > 4000 else err)
                results.append(sr)
                continue
            print("pip install: OK")
        else:
            sr.pip_ok = True
            print("pip install: skipped (--skip-install)")

        code, combined = run_test_script(venv_py, repo_root, sphere_script)
        sr.sphere_exit = code
        sr.sphere_ok = code == 0
        sr.sphere_error = combined if code != 0 else ""
        status: Literal["OK", "FAIL"] = "OK" if code == 0 else "FAIL"
        print(f"flow_past_sphere_3d_test exit code: {code} ({status})")
        if code != 0 and combined:
            print("--- test output (tail) ---\n", combined[-2500:] if len(combined) > 2500 else combined)

        results.append(sr)

    # ------------------------------------------------------------------ summary
    print("\n")
    print("=" * 72)
    print("FULL SUMMARY — XLB install matrix + flow_past_sphere_3d_test.py")
    print("=" * 72)
    print(f"Repository: {repo_root}")
    print(f"Venv root:  {venv_root}")
    print()

    col_w = max(len(r.name) for r in results) if results else 12
    hdr = f"{'Scenario':<{col_w}}  {'pip':^5}  {'3d':^5}"
    print(hdr)
    print("-" * len(hdr))

    any_test_failed = False
    any_pip_failed = False

    for r in results:
        pip_s = "ok" if r.pip_ok else "FAIL"
        if not r.pip_ok:
            any_pip_failed = True
        if r.sphere_ok is None:
            sph_s = "—"
        elif r.sphere_ok:
            sph_s = "ok"
        else:
            sph_s = "FAIL"
            any_test_failed = True
        print(f"{r.name:<{col_w}}  {pip_s:^5}  {sph_s:^5}")
        print(f"{'':<{col_w}}          {r.label}")

    print()
    print("Details per scenario")
    print("-" * 72)
    for r in results:
        print(f"\n* {r.name} — {r.label}")
        print(f"  pip ok: {r.pip_ok}")
        if r.pip_error and not r.pip_ok:
            print(f"  pip error (truncated): {r.pip_error[:1500]}...")
        print(f"  sphere test exit: {r.sphere_exit}")
        if r.sphere_error and r.sphere_exit not in (0, None):
            print(f"  sphere output (truncated): {r.sphere_error[:2000]}...")

    print()
    print("=" * 72)
    if any_pip_failed:
        print("Overall: FAILURE (at least one pip install failed).")
        return 1
    if any_test_failed:
        print("Overall: FAILURE (at least one 3D smoke test returned non-zero).")
        return 1
    print("Overall: SUCCESS — all configured installs and tests completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

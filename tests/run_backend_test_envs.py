#!/usr/bin/env python3
"""
Build a fresh virtualenv per compute backend, install XLB from the repository
root, and run the test suite in each one.

Usage (from the repository root)::

    python tests/run_backend_test_envs.py                 # warp and neon
    python tests/run_backend_test_envs.py --backend neon  # just one
    python tests/run_backend_test_envs.py --reuse         # keep existing venvs
    python tests/run_backend_test_envs.py --jax cuda      # GPU jaxlib in both envs

Profiles
--------
warp
    ``pip install -e .[warp,test]``. Warp comes from PyPI (``warp-lang``). The
    suite's WARP and JAX tests run; the Neon tests in ``tests/backends`` skip
    themselves because ``neon`` is not importable.
neon
    ``pip install -e .[neon,test]``. Installs the ``neon_gpu`` wheel, which
    bundles its own Warp fork, and XLB's setup hook removes ``warp-lang`` so
    that fork is the only Warp on the path. The same WARP and JAX tests
    therefore run *against Neon's bundled Warp*, and the Neon tests also run.

JAX is a core dependency (``install_requires``), so both profiles get it
regardless of the backend; ``--jax cuda`` adds setup.py's ``cuda`` extra to
swap the CPU-only jaxlib for a GPU one.

Options
-------
--backend {warp,neon,both}  Profiles to build and test (default: both).
--jax {cpu,cuda}            JAX flavor to install in every env (default: cpu).
--venv-root PATH            Where to create the venvs
                            (default: .xlb_backend_test_venvs, gitignored).
--python PATH               Interpreter used to create the venvs. The [neon]
                            extra resolves a wheel matching its version, so use
                            this to test another CPython (default: this one).
--reuse                     Reuse existing venvs instead of recreating them.
--skip-install              Only run tests, assuming the venvs are populated.
--skip-install-tests        Skip tests/install, whose Neon cleanup test builds
                            another venv and downloads the neon_gpu wheel.
--tests PATH [PATH ...]     Test paths to run (default: tests).
--pytest-args "ARGS"        Extra pytest flags. Paths belong in --tests; these
                            are appended, so a path here widens the selection.

Environment
-----------
XLB_BACKEND_VENV_ROOT       Same as --venv-root.

Exits non-zero if any install or test run fails.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# pytest reports "no tests collected" rather than success when every selected
# test is skipped, which is what --tests tests/backends does in the warp env,
# where the Neon tests skip themselves. Not a failure.
PYTEST_NO_TESTS_COLLECTED = 5

PROFILES: dict[str, dict[str, str]] = {
    "warp": {
        "extras": "warp,test",
        "summary": "Warp backend, warp-lang from PyPI",
    },
    "neon": {
        "extras": "neon,test",
        "summary": "Neon backend, neon_gpu wheel with its bundled Warp fork",
    },
}


@dataclass
class ProfileResult:
    name: str
    extras: str
    install_ok: bool = False
    install_output: str = ""
    tests_ok: bool | None = None
    tests_exit: int | None = None
    stack: dict[str, str] = field(default_factory=dict)
    warning: str = ""
    seconds: float = 0.0


def venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def create_venv(venv_dir: Path, base_python: str, *, reuse: bool) -> Path:
    py = venv_python(venv_dir)
    if reuse and py.is_file():
        print(f"venv: reusing {venv_dir}")
        return py
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"venv: creating {venv_dir} with {base_python}")
    subprocess.run([base_python, "-m", "venv", str(venv_dir)], check=True)
    return py


def install_xlb(py: Path, repo_root: Path, extras: str) -> tuple[bool, str]:
    """Editable-install XLB from *repo_root* with *extras*, capturing output."""
    steps = [
        [str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel"],
        [str(py), "-m", "pip", "install", "-e", f".[{extras}]"],
    ]
    transcript = ""
    for cmd in steps:
        print(f"pip: {' '.join(cmd[3:])}")
        proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
        transcript += proc.stdout + proc.stderr
        if proc.returncode != 0:
            return False, transcript
    return True, transcript


def describe_stack(py: Path, repo_root: Path) -> dict[str, str]:
    """Report which Warp, Neon and JAX the env actually resolved to.

    A wheel's version string alone does not identify Neon's Warp fork, so the
    install location is reported too.
    """
    probe = """
import importlib.util, json, os
out = {}

def where(mod):
    spec = importlib.util.find_spec(mod)
    return os.path.dirname(spec.origin) if spec and spec.origin else ""

try:
    import warp
    out["warp"] = warp.__version__
    out["warp_path"] = where("warp")
except Exception as exc:
    out["warp"] = f"unavailable ({exc})"

out["neon"] = "importable" if importlib.util.find_spec("neon") else "absent"
out["warp_lang_dist"] = "absent"
try:
    from importlib.metadata import version
    out["warp_lang_dist"] = version("warp-lang")
except Exception:
    pass

try:
    import jax
    out["jax"] = jax.__version__
    out["jax_devices"] = ", ".join(sorted({d.platform for d in jax.devices()}))
except Exception as exc:
    out["jax"] = f"unavailable ({exc})"

print(json.dumps(out))
"""
    proc = subprocess.run([str(py), "-c", probe], cwd=repo_root, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout).strip()[-400:]}
    import json

    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"error": "could not parse environment probe output"}


def warn_on_mixed_warp(stack: dict[str, str]) -> str:
    """Flag a Neon env that also has the ``warp-lang`` distribution registered.

    ``neon_gpu`` ships its own Warp fork at the same ``site-packages/warp``
    path. XLB's ``[neon]`` extra no longer pulls ``warp-lang`` in, but pip does
    not remove one that was installed beforehand, so a reused venv can still
    hold both. Whichever pip unpacked last owns the files, and a later
    ``pip uninstall warp-lang`` would delete files Neon needs. Tests pass either
    way, so this is only detectable by looking.
    """
    if stack.get("neon") != "importable" or stack.get("warp_lang_dist") == "absent":
        return ""
    message = (
        f"WARNING: both neon_gpu and warp-lang {stack['warp_lang_dist']} are installed.\n"
        f"         The imported warp is {stack.get('warp', '?')}, so Neon's fork currently owns\n"
        f"         site-packages/warp, but the env is order-dependent: uninstalling or\n"
        f"         upgrading warp-lang would overwrite or delete Neon's Warp."
    )
    print(message)
    return message


def run_pytest(py: Path, repo_root: Path, *, targets: list[str], skip_install_tests: bool, extra_args: list[str]) -> int:
    """Run the suite with output streamed, so progress is visible."""
    cmd = [str(py), "-m", "pytest", *targets, "-q"]
    if skip_install_tests:
        cmd += ["--ignore", "tests/install"]
    cmd += extra_args
    print(f"pytest: {' '.join(cmd[1:])}\n")
    return subprocess.run(cmd, cwd=repo_root, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a venv per backend, install XLB from the repo root, and run the tests.",
    )
    parser.add_argument("--backend", choices=[*PROFILES, "both"], default="both")
    parser.add_argument("--jax", choices=["cpu", "cuda"], default="cpu", help="JAX flavor for every env")
    parser.add_argument(
        "--venv-root",
        type=Path,
        default=None,
        help="Directory holding the per-backend venvs",
    )
    parser.add_argument("--python", default=sys.executable, help="Interpreter used to create the venvs")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reuse", action="store_true", help="Reuse existing venvs")
    parser.add_argument("--skip-install", action="store_true", help="Run tests only")
    parser.add_argument("--skip-install-tests", action="store_true", help="Skip tests/install")
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["tests"],
        metavar="PATH",
        help="Test paths to run (default: tests)",
    )
    parser.add_argument("--pytest-args", default="", help="Extra pytest flags; use --tests to narrow paths")
    args = parser.parse_args()

    # Subprocesses inherit this stdout and write straight to it, so keep our own
    # prints line-buffered or they would surface out of order once redirected.
    sys.stdout.reconfigure(line_buffering=True)

    repo_root: Path = args.repo_root.resolve()
    if not (repo_root / "setup.py").is_file():
        print(f"ERROR: no setup.py in {repo_root}; pass --repo-root", file=sys.stderr)
        return 2

    default_root = os.environ.get("XLB_BACKEND_VENV_ROOT", repo_root / ".xlb_backend_test_venvs")
    venv_root = Path(args.venv_root or default_root).resolve()
    selected = list(PROFILES) if args.backend == "both" else [args.backend]
    extra_pytest = shlex.split(args.pytest_args)

    results: list[ProfileResult] = []
    for name in selected:
        extras = PROFILES[name]["extras"]
        if args.jax == "cuda":
            extras += ",cuda"
        result = ProfileResult(name=name, extras=extras)
        started = time.time()

        print(f"\n{'=' * 78}")
        print(f"Profile: {name} — {PROFILES[name]['summary']}")
        print(f"Install: pip install -e .[{extras}]")
        print(f"{'=' * 78}")

        try:
            py = create_venv(venv_root / name, args.python, reuse=args.reuse or args.skip_install)
        except (subprocess.CalledProcessError, OSError) as exc:
            result.install_output = f"venv creation failed: {exc}"
            print(result.install_output, file=sys.stderr)
            results.append(result)
            continue

        if args.skip_install:
            result.install_ok = True
            print("pip: skipped (--skip-install)")
        else:
            result.install_ok, result.install_output = install_xlb(py, repo_root, extras)
            if not result.install_ok:
                print("PIP INSTALL FAILED\n" + result.install_output[-4000:], file=sys.stderr)
                result.seconds = time.time() - started
                results.append(result)
                continue
            print("pip: ok")

        result.stack = describe_stack(py, repo_root)
        for key, value in result.stack.items():
            print(f"  {key}: {value}")
        result.warning = warn_on_mixed_warp(result.stack)
        print()

        result.tests_exit = run_pytest(
            py,
            repo_root,
            targets=args.tests,
            skip_install_tests=args.skip_install_tests,
            extra_args=extra_pytest,
        )
        if result.tests_exit == PYTEST_NO_TESTS_COLLECTED:
            result.tests_ok = None
            verdict = "no tests ran (all skipped for this profile)"
        else:
            result.tests_ok = result.tests_exit == 0
            verdict = "OK" if result.tests_ok else "FAIL"
        result.seconds = time.time() - started
        print(f"\npytest exit code: {result.tests_exit} ({verdict})")
        results.append(result)

    print(f"\n{'=' * 78}")
    print("SUMMARY — XLB per-backend environments")
    print(f"{'=' * 78}")
    print(f"Repository: {repo_root}")
    print(f"Venv root:  {venv_root}\n")

    width = max((len(r.name) for r in results), default=8)
    print(f"{'profile':<{width}}  {'install':^8}  {'tests':^6}  {'minutes':^8}  stack")
    print("-" * 78)
    for r in results:
        install_s = "ok" if r.install_ok else "FAIL"
        tests_s = "—" if r.tests_ok is None else ("ok" if r.tests_ok else "FAIL")
        warp_desc = r.stack.get("warp", "?")
        if r.stack.get("neon") == "importable":
            warp_desc += " (neon-bundled)" if r.stack.get("warp_lang_dist") == "absent" else " (+warp-lang!)"
        print(f"{r.name:<{width}}  {install_s:^8}  {tests_s:^6}  {r.seconds / 60:^8.1f}  warp {warp_desc}, jax {r.stack.get('jax', '?')}")

    warned = [r for r in results if r.warning]
    if warned:
        print()
        for r in warned:
            print(f"{r.name}:\n{r.warning}")

    failed = [r for r in results if not r.install_ok or r.tests_ok is False]
    print()
    if failed:
        print(f"Overall: FAILURE ({', '.join(r.name for r in failed)})")
        return 1
    print("Overall: SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

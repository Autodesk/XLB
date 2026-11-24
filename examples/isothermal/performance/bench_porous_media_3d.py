"""Benchmark single-component MRT flow through a reproducible porous medium.

The setup uses the porous-media collision matrix, EOS, relaxation values, and
wetting parameters on a configurable synthetic geometry.

Run directly for A/B numbers (requires a GPU):
    python tests/performance/bench_porous_media_3d.py [--nx 64] [--steps 200]
"""

import os

# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
# os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer=").strip()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jax_lab.core.boundary_conditions import BounceBack, ExactNonEquilibriumExtrapolation  # noqa: E402
from jax_lab.core.shanchen.eos import PengRobinson  # noqa: E402
from jax_lab.core.lattice import LatticeD3Q19  # noqa: E402
from jax_lab.core.shanchen.multiphase import MultiphaseMRT  # noqa: E402

RHO_L = 6.499210784
RHO_G = 0.379598891
S2 = 0.8
THETA_W = np.pi / 3
PHI_W = 1.15
DELTA_RHO_W = 0.0
PRECISION = "f32/f32"


def require_gpu():
    devices = jax.devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"GPU test required, found: {devices}")


def _mrt_matrix_d3q19():
    e = np.asarray(LatticeD3Q19().c).T
    ex, ey, ez = e[:, 0], e[:, 1], e[:, 2]
    M = np.zeros((19, 19))
    M[0, :] = ex**0
    M[1, :] = ex
    M[2, :] = ey
    M[3, :] = ez
    M[4, :] = ex * ey
    M[5, :] = ex * ez
    M[6, :] = ey * ez
    M[7, :] = ex * ex
    M[8, :] = ey * ey
    M[9, :] = ez * ez
    M[10, :] = ex * ey * ey
    M[11, :] = ex * ez * ez
    M[12, :] = ey * ex * ex
    M[13, :] = ez * ex * ex
    M[14, :] = ey * ez * ez
    M[15, :] = ez * ey * ey
    M[16, :] = ex * ex * ey * ey
    M[17, :] = ex * ex * ez * ez
    M[18, :] = ey * ey * ez * ez
    return M


def _random_porous_mask(nx, ny, nz, seed, solid_fraction=0.35):
    """Generate a reproducible porous mask with connected solid regions."""
    rng = np.random.default_rng(seed)
    noise = rng.random((nx, ny, nz))
    # Cheap separable box blur (3 passes) instead of a scipy dependency, to correlate neighboring voxels.
    for axis in range(3):
        noise = (noise + np.roll(noise, 1, axis=axis) + np.roll(noise, -1, axis=axis)) / 3.0
    threshold = np.quantile(noise, solid_fraction)
    mask = noise < threshold
    mask[:, :, :2] = False  # keep a clear inlet/outlet slab at each z end
    mask[:, :, -2:] = False
    return mask


class PorousMediaBenchmark(MultiphaseMRT):
    def __init__(self, mask, **kwargs):
        self._mask = mask
        super().__init__(**kwargs)

    def initialize_macroscopic_fields(self):
        z = np.arange(self.nz).reshape(1, 1, self.nz)
        L = self.nz // 3
        width = 3
        rho = 0.5 * (RHO_L + RHO_G) - 0.5 * (RHO_L - RHO_G) * np.tanh(2 * (z - L) / width)
        rho = np.broadcast_to(rho, (self.nx, self.ny, self.nz)).copy().reshape(self.nx, self.ny, self.nz, 1)
        rho = self.distributed_array_init((self.nx, self.ny, self.nz, 1), self.precision_policy.compute_dtype, init_val=rho)
        rho = self.precision_policy.cast_to_output(rho)

        u = np.zeros((self.nx, self.ny, self.nz, 3))
        u = self.distributed_array_init((self.nx, self.ny, self.nz, 3), self.precision_policy.compute_dtype, init_val=u)
        u = self.precision_policy.cast_to_output(u)
        return [rho], [u]

    def set_boundary_conditions(self):
        bottom = self.bounding_box_indices["bottom"]
        self.BCs[0].append(BounceBack(tuple(bottom.T), self.grid_info, self.precision_policy))

        top = self.bounding_box_indices["top"]
        # 0.75*RHO_G (a stronger evaporation-like forcing, matching the reference script) goes unstable by
        # ~step 30-40 on this synthetic random geometry; 0.93*RHO_G stays finite past 100 steps at every size
        # tested (64^3-192^3), so it's used here purely to give this benchmark a stable multi-hundred-step
        # window - not a claim about what's stable for a real porous-media study.
        prescribed_rho = 0.93 * RHO_G * jnp.ones((top.shape[0], 1))
        self.BCs[0].append(ExactNonEquilibriumExtrapolation(tuple(top.T), self.grid_info, self.precision_policy, prescribed_rho, "density"))

        porous_indices = np.array(np.where(self._mask)).T
        self.BCs[0].append(BounceBack(tuple(porous_indices.T), self.grid_info, self.precision_policy, THETA_W, PHI_W, DELTA_RHO_W))


def build_simulation(nx, ny, nz, seed, wetting_formulation="improved_virtual_density"):
    n_devices = jax.device_count()
    if nx % n_devices:
        nx += n_devices - nx % n_devices
    mask = _random_porous_mask(nx, ny, nz, seed)

    s_0, s_1, s_b, s_2, s_4 = [1.0], [1.0], [0.8], [S2], [1.0]
    s_3 = [(16 - 8 * S2) / (8 - S2)]
    kwargs = {
        "n_components": 1,
        "lattice": LatticeD3Q19(PRECISION),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "g_kkprime": -np.ones((1, 1)),
        "lambda_": [1.0],
        "EOS": PengRobinson(a=[3 / 49], b=[2 / 21], pr_omega=[0.344], R=[1.0], T=0.86 * 0.1093785558),
        "M": [_mrt_matrix_d3q19()],
        "s_0": s_0,
        "s_1": s_1,
        "s_b": s_b,
        "s_2": s_2,
        "s_3": s_3,
        "s_4": s_4,
        "kappa": [0.0],
        "wetting_formulation": wetting_formulation,
        "s_rho": s_0,
        "s_e": s_b,
        "s_eta": s_b,
        "s_j": s_1,
        "s_q": s_3,
        "s_v": s_2,
        "s_pi": s_3,
        "s_m": s_4,
        "precision": PRECISION,
        "io_rate": 0,
        "print_info_rate": 0,
        "compute_MLUPS": True,
        "checkpoint_rate": 0,
    }
    return PorousMediaBenchmark(mask, **kwargs)


def run_benchmark(
    nx=64,
    ny=64,
    nz=64,
    steps=200,
    seed=0,
    sample_memory=False,
    wetting_formulation="improved_virtual_density",
    save_arrays=True,
):
    """Measure throughput and optionally sample live device memory each step.

    Per-step memory sampling synchronizes the device, so that run's MLUPS is not
    representative of normal throughput.
    """
    require_gpu()
    sim = build_simulation(nx, ny, nz, seed, wetting_formulation)

    state_tree = sim.state_from_populations(sim.assign_fields_sharded())
    state_tree, _ = sim.step(state_tree, 0)  # compile, excluded from timing
    jax.block_until_ready(state_tree)

    device = jax.local_devices()[0]
    memory_before = device.memory_stats() or {}
    memory_trace = []

    start = time.time()
    for timestep in range(1, steps + 1):
        state_tree, _ = sim.step(state_tree, timestep)
        if sample_memory:
            jax.block_until_ready(state_tree)
            memory_trace.append((device.memory_stats() or {}).get("bytes_in_use"))
    jax.block_until_ready(state_tree)
    elapsed = time.time() - start
    f_tree = sim.poststreaming(state_tree)

    memory_after = device.memory_stats() or {}
    rho_tree, _ = sim.update_macroscopic(f_tree)

    rho = np.asarray(rho_tree[0])
    voxels = sim.nx * sim.ny * sim.nz
    result = {
        "nx": sim.nx,
        "ny": sim.ny,
        "nz": sim.nz,
        "n_devices": jax.device_count(),
        "steps": steps,
        "wetting_formulation": wetting_formulation,
        "solid_fraction": float(np.mean(sim._mask)),
        "elapsed_seconds": elapsed,
        "mlups": voxels * steps / elapsed / 1e6,
        "bytes_in_use": memory_after.get("bytes_in_use"),
        "peak_bytes_in_use": memory_after.get("peak_bytes_in_use", memory_before.get("peak_bytes_in_use")),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "rho_total_mass": float(rho.sum()),
        "rho_finite": bool(np.isfinite(rho).all()),
    }
    if save_arrays:
        result["population"] = np.asarray(f_tree[0])
        result["force"] = np.asarray(sim.compute_force(rho_tree)[0])
    if sample_memory:
        result["bytes_in_use_min"] = min(memory_trace)
        result["bytes_in_use_max"] = max(memory_trace)
        result["bytes_in_use_mean"] = sum(memory_trace) / len(memory_trace)
        result["bytes_in_use_trace"] = memory_trace
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wetting-formulation",
        choices=("improved_virtual_density", "geometric"),
        default="improved_virtual_density",
    )
    parser.add_argument("--out", type=str, default="bench_result")
    parser.add_argument("--sample-memory", action="store_true", help="Trace bytes_in_use every step (see run_benchmark)")
    parser.add_argument("--summary-only", action="store_true", help="Skip full population/force array copies and saves.")
    args = parser.parse_args()

    result = run_benchmark(
        args.nx,
        args.ny,
        args.nz,
        args.steps,
        args.seed,
        sample_memory=args.sample_memory,
        wetting_formulation=args.wetting_formulation,
        save_arrays=not args.summary_only,
    )
    population = result.pop("population", None)
    force = result.pop("force", None)
    trace = result.pop("bytes_in_use_trace", None)

    print(json.dumps(result, indent=2))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    if population is not None:
        np.save(out_dir / "population.npy", population)
        np.save(out_dir / "force.npy", force)
    if trace is not None:
        np.save(out_dir / "bytes_in_use_trace.npy", np.asarray(trace))
    print(f"Saved arrays and summary to {out_dir}/")


if __name__ == "__main__":
    main()

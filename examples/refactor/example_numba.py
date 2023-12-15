# from IPython import display
import numpy as np
import cupy as cp
import scipy
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import cuda, config

import xlb

config.CUDA_ARRAY_INTERFACE_SYNC = False

if __name__ == "__main__":
    # XLB precision policy
    precision_policy = xlb.precision_policy.Fp32Fp32()

    # XLB lattice
    lattice = xlb.lattice.D3Q19()

    # XLB collision model
    collision = xlb.collision.BGK()

    # Make XLB compute kernels
    compute = xlb.compute_constructor.NumbaConstructor(
        lattice=lattice,
        collision=collision,
        boundary_conditions=[],
        forcing=None,
        precision_policy=precision_policy,
    )

    # Make taylor green vortex initial condition
    tau = 0.505
    vel = 0.1 * 1.0 / np.sqrt(3.0)
    nr = 256
    lin = cp.linspace(0, 2 * np.pi, nr, endpoint=False, dtype=cp.float32)
    X, Y, Z = cp.meshgrid(lin, lin, lin, indexing="ij")
    X = X[None, ...]
    Y = Y[None, ...]
    Z = Z[None, ...]
    u = vel * cp.sin(X) * cp.cos(Y) * cp.cos(Z)
    v = -vel * cp.cos(X) * cp.sin(Y) * cp.cos(Z)
    w = cp.zeros_like(X)
    rho = (
        3.0
        * vel**2
        * (1.0 / 16.0)
        * (cp.cos(2 * X) + cp.cos(2 * Y) + cp.cos(2 * Z))
        + 1.0)
    u = cp.concatenate([u, v, w], axis=-1)

    # Allocate f
    f0 = cp.zeros((19, nr, nr, nr), dtype=cp.float32)
    f1 = cp.zeros((19, nr, nr, nr), dtype=cp.float32)

    # Get f from u, rho
    compute.equilibrium(rho, u, f0)

    # Run compute kernel on f
    tic = time.time()
    nr_iter = 128
    for i in tqdm(range(nr_iter)):
        compute.step(f0, f1, i)
        f0, f1 = f1, f0

        if i % 4 == 0:
            ## Get u, rho from f
            #rho, u = compute.macroscopic(f)
            #norm_u = jnp.linalg.norm(u, axis=-1)

            # Plot
            plt.imsave(f"img_{str(i).zfill(5)}.png", f0[8, nr // 2, :, :].get(), cmap="jet")

    # Sync to host
    cp.cuda.stream.get_current_stream().synchronize()
    toc = time.time()
    print(f"MLUPS: {(nr_iter * nr**3) / (toc - tic) / 1e6}")

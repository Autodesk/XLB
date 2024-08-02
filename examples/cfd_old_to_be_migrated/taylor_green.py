# Simple Taylor green example using the functional interface to xlb

import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from functools import partial
from typing import Any
import jax.numpy as jnp
from jax import jit
import warp as wp

wp.init()

import xlb
from xlb.operator import Operator


class TaylorGreenInitializer(Operator):
    """
    Initialize the Taylor-Green vortex.
    """

    @Operator.register_backend(xlb.ComputeBackend.JAX)
    # @partial(jit, static_argnums=(0))
    def jax_implementation(self, vel, nr):
        # Make meshgrid
        x = jnp.linspace(0, 2 * jnp.pi, nr)
        y = jnp.linspace(0, 2 * jnp.pi, nr)
        z = jnp.linspace(0, 2 * jnp.pi, nr)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

        # Compute u
        u = jnp.stack(
            [
                vel * jnp.sin(X) * jnp.cos(Y) * jnp.cos(Z),
                -vel * jnp.cos(X) * jnp.sin(Y) * jnp.cos(Z),
                jnp.zeros_like(X),
            ],
            axis=0,
        )

        # Compute rho
        rho = 3.0 * vel * vel * (1.0 / 16.0) * (jnp.cos(2.0 * X) + (jnp.cos(2.0 * Y) * (jnp.cos(2.0 * Z) + 2.0))) + 1.0
        rho = jnp.expand_dims(rho, axis=0)

        return rho, u

    def _construct_warp(self):
        # Construct the warp kernel
        @wp.kernel
        def kernel(
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            vel: float,
            nr: int,
        ):
            # Get the global index
            i, j, k = wp.tid()

            # Get real pos
            x = 2.0 * wp.pi * wp.float(i) / wp.float(nr)
            y = 2.0 * wp.pi * wp.float(j) / wp.float(nr)
            z = 2.0 * wp.pi * wp.float(k) / wp.float(nr)

            # Compute u
            u[0, i, j, k] = vel * wp.sin(x) * wp.cos(y) * wp.cos(z)
            u[1, i, j, k] = -vel * wp.cos(x) * wp.sin(y) * wp.cos(z)
            u[2, i, j, k] = 0.0

            # Compute rho
            rho[0, i, j, k] = 3.0 * vel * vel * (1.0 / 16.0) * (wp.cos(2.0 * x) + (wp.cos(2.0 * y) * (wp.cos(2.0 * z) + 2.0))) + 1.0

        return None, kernel

    @Operator.register_backend(xlb.ComputeBackend.WARP)
    def warp_implementation(self, rho, u, vel, nr):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                rho,
                u,
                vel,
                nr,
            ],
            dim=rho.shape[1:],
        )
        return rho, u


def run_taylor_green(backend, compute_mlup=True):
    # Set the compute backend
    if backend == "warp":
        compute_backend = xlb.ComputeBackend.WARP
    elif backend == "jax":
        compute_backend = xlb.ComputeBackend.JAX

    # Set the precision policy
    precision_policy = xlb.PrecisionPolicy.FP32FP32

    # Set the velocity set
    velocity_set = xlb.velocity_set.D3Q19()

    # Make grid
    nr = 128
    shape = (nr, nr, nr)
    if backend == "jax":
        grid = xlb.grid.JaxGrid(shape=shape)
    elif backend == "warp":
        grid = xlb.grid.WarpGrid(shape=shape)

    # Make feilds
    rho = grid.create_field(cardinality=1, precision=xlb.Precision.FP32)
    u = grid.create_field(cardinality=velocity_set.d, precision=xlb.Precision.FP32)
    f0 = grid.create_field(cardinality=velocity_set.q, precision=xlb.Precision.FP32)
    f1 = grid.create_field(cardinality=velocity_set.q, precision=xlb.Precision.FP32)
    boundary_mask = grid.create_field(cardinality=1, precision=xlb.Precision.UINT8)
    missing_mask = grid.create_field(cardinality=velocity_set.q, precision=xlb.Precision.BOOL)

    # Make operators
    initializer = TaylorGreenInitializer(velocity_set=velocity_set, precision_policy=precision_policy, compute_backend=compute_backend)
    collision = xlb.operator.collision.BGK(omega=1.9, velocity_set=velocity_set, precision_policy=precision_policy, compute_backend=compute_backend)
    equilibrium = xlb.operator.equilibrium.QuadraticEquilibrium(
        velocity_set=velocity_set, precision_policy=precision_policy, compute_backend=compute_backend
    )
    macroscopic = xlb.operator.macroscopic.Macroscopic(velocity_set=velocity_set, precision_policy=precision_policy, compute_backend=compute_backend)
    stream = xlb.operator.stream.Stream(velocity_set=velocity_set, precision_policy=precision_policy, compute_backend=compute_backend)
    stepper = xlb.operator.stepper.IncompressibleNavierStokesStepper(
        collision=collision, equilibrium=equilibrium, macroscopic=macroscopic, stream=stream
    )

    # Parrallelize the stepper TODO: Add this functionality
    # stepper = grid.parallelize_operator(stepper)

    # Set initial conditions
    if backend == "warp":
        rho, u = initializer(rho, u, 0.1, nr)
        f0 = equilibrium(rho, u, f0)
    elif backend == "jax":
        rho, u = initializer(0.1, nr)
        f0 = equilibrium(rho, u)

    # Time stepping
    plot_freq = 32
    save_dir = "taylor_green"
    os.makedirs(save_dir, exist_ok=True)
    num_steps = 8192
    start = time.time()

    for _ in tqdm(range(num_steps)):
        # Time step
        if backend == "warp":
            f1 = stepper(f0, f1, boundary_mask, missing_mask, _)
            f1, f0 = f0, f1
        elif backend == "jax":
            f0 = stepper(f0, boundary_mask, missing_mask, _)

        # Plot if needed
        if (_ % plot_freq == 0) and (not compute_mlup):
            if backend == "warp":
                rho, u = macroscopic(f0, rho, u)
                local_u = u.numpy()
            elif backend == "jax":
                rho, local_u = macroscopic(f0)

            plt.imshow(local_u[0, :, nr // 2, :])
            plt.colorbar()
            plt.savefig(f"{save_dir}/{str(_).zfill(6)}.png")
            plt.close()
    wp.synchronize()
    end = time.time()

    # Print MLUPS
    print(f"MLUPS: {num_steps * nr**3 / (end - start) / 1e6}")


if __name__ == "__main__":
    # Run Taylor-Green vortex on different backends
    backends = ["warp", "jax"]
    # backends = ["jax"]
    for backend in backends:
        run_taylor_green(backend, compute_mlup=True)

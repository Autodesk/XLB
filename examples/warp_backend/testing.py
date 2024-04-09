# from IPython import display
import numpy as np
import jax
import jax.numpy as jnp
import scipy
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import warp as wp
wp.init()

import xlb


def test_backends(compute_backend):

    # Set parameters
    precision_policy = xlb.PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D3Q27()

    # Make operators
    collision = xlb.operator.collision.BGK(
            omega=1.0,
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend)
    equilibrium = xlb.operator.equilibrium.QuadraticEquilibrium(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend)
    macroscopic = xlb.operator.macroscopic.Macroscopic(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend)
    stream = xlb.operator.stream.Stream(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend)
    bounceback = xlb.operator.boundary_condition.FullBounceBack.from_indices(
            indices=np.array([[0, 0, 0], [0, 0, 1]]),
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend)
    stepper = xlb.operator.stepper.IncompressibleNavierStokesStepper(
            collision=collision,
            equilibrium=equilibrium,
            macroscopic=macroscopic,
            stream=stream,
            boundary_conditions=[bounceback])

    # Test operators
    if compute_backend == xlb.ComputeBackend.WARP:
        # Make warp arrays
        nr = 128
        f_0 = wp.zeros((27, nr, nr, nr), dtype=wp.float32)
        f_1 = wp.zeros((27, nr, nr, nr), dtype=wp.float32)
        f_out = wp.zeros((27, nr, nr, nr), dtype=wp.float32)
        u = wp.zeros((3, nr, nr, nr), dtype=wp.float32)
        rho = wp.zeros((1, nr, nr, nr), dtype=wp.float32)
        boundary_id = wp.zeros((1, nr, nr, nr), dtype=wp.uint8)
        boundary = wp.zeros((1, nr, nr, nr), dtype=wp.bool)
        mask = wp.zeros((27, nr, nr, nr), dtype=wp.bool)

        # Test operators
        collision(f_0, f_1, rho, u, f_out)
        equilibrium(rho, u, f_0)
        macroscopic(f_0, rho, u)
        stream(f_0, f_1)
        bounceback(f_0, f_1, f_out, boundary, mask)
        #bounceback.boundary_masker((0, 0, 0), boundary_id, mask, 1)



    elif compute_backend == xlb.ComputeBackend.JAX:
        # Make jax arrays
        nr = 128
        f_0 = jnp.zeros((27, nr, nr, nr), dtype=jnp.float32)
        f_1 = jnp.zeros((27, nr, nr, nr), dtype=jnp.float32)
        f_out = jnp.zeros((27, nr, nr, nr), dtype=jnp.float32)
        u = jnp.zeros((3, nr, nr, nr), dtype=jnp.float32)
        rho = jnp.zeros((1, nr, nr, nr), dtype=jnp.float32)
        boundary_id = jnp.zeros((1, nr, nr, nr), dtype=jnp.uint8)
        boundary = jnp.zeros((1, nr, nr, nr), dtype=jnp.bool_)
        mask = jnp.zeros((27, nr, nr, nr), dtype=jnp.bool_)

        # Test operators
        collision(f_0, f_1, rho, u)
        equilibrium(rho, u)
        macroscopic(f_0)
        stream(f_0)
        bounceback(f_0, f_1, boundary, mask)
        bounceback.boundary_masker((0, 0, 0), boundary_id, mask, 1)
        stepper(f_0, boundary_id, mask, 0)



if __name__ == "__main__":

    # Test backends
    compute_backend = [
        xlb.ComputeBackend.WARP,
        xlb.ComputeBackend.JAX
    ]

    for compute_backend in compute_backend:
        test_backends(compute_backend)
        print(f"Backend {compute_backend} passed all tests.")

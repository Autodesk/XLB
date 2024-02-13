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

if __name__ == "__main__":

    # Make operator
    precision_policy = xlb.PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D3Q27()
    compute_backend = xlb.ComputeBackend.WARP
    equilibrium = xlb.operator.equilibrium.QuadraticEquilibrium(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend)
    macroscopic = xlb.operator.macroscopic.Macroscopic(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend)

    # Make warp arrays
    nr = 128
    f = wp.zeros((27, nr, nr, nr), dtype=wp.float32)
    u = wp.zeros((3, nr, nr, nr), dtype=wp.float32)
    rho = wp.zeros((1, nr, nr, nr), dtype=wp.float32)

    # Run simulation
    equilibrium(rho, u, f)
    macroscopic(f, rho, u)

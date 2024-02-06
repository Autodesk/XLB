# Description: This file contains a simple example of using the OOCmap
# decorator to apply a function to a distributed array.
# Solves Lattice Boltzmann Taylor Green vortex decay

import time
import warp as wp
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cupy as cp
import time
from tqdm import tqdm
from numba import cuda
import numba
import math
import jax.numpy as jnp
import jax
from jax import jit
from functools import partial

# Initialize Warp
wp.init()

@wp.func
def warp_set_f(
    f: wp.array4d(dtype=float),
    value: float,
    q: int,
    i: int,
    j: int,
    k: int,
    width: int,
    height: int,
    length: int,
):
    # Modulo
    if i < 0:
        i += width
    if j < 0:
        j += height
    if k < 0:
        k += length
    if i >= width:
        i -= width
    if j >= height:
        j -= height
    if k >= length:
        k -= length
    f[q, i, j, k] = value

@wp.kernel
def warp_collide_stream(
    f0: wp.array4d(dtype=float),
    f1: wp.array4d(dtype=float),
    width: int,
    height: int,
    length: int,
    tau: float,
):

    # get index
    x, y, z = wp.tid()

    # sample needed points
    f_1_1_1 = f0[0, x, y, z]
    f_2_1_1 = f0[1, x, y, z]
    f_0_1_1 = f0[2, x, y, z]
    f_1_2_1 = f0[3, x, y, z]
    f_1_0_1 = f0[4, x, y, z]
    f_1_1_2 = f0[5, x, y, z]
    f_1_1_0 = f0[6, x, y, z]
    f_1_2_2 = f0[7, x, y, z]
    f_1_0_0 = f0[8, x, y, z]
    f_1_2_0 = f0[9, x, y, z]
    f_1_0_2 = f0[10, x, y, z]
    f_2_1_2 = f0[11, x, y, z]
    f_0_1_0 = f0[12, x, y, z]
    f_2_1_0 = f0[13, x, y, z]
    f_0_1_2 = f0[14, x, y, z]
    f_2_2_1 = f0[15, x, y, z]
    f_0_0_1 = f0[16, x, y, z]
    f_2_0_1 = f0[17, x, y, z]
    f_0_2_1 = f0[18, x, y, z]

    # compute u and p
    p = (f_1_1_1
       + f_2_1_1 + f_0_1_1
       + f_1_2_1 + f_1_0_1
       + f_1_1_2 + f_1_1_0
       + f_1_2_2 + f_1_0_0
       + f_1_2_0 + f_1_0_2
       + f_2_1_2 + f_0_1_0
       + f_2_1_0 + f_0_1_2
       + f_2_2_1 + f_0_0_1
       + f_2_0_1 + f_0_2_1)
    u = (f_2_1_1 - f_0_1_1
       + f_2_1_2 - f_0_1_0
       + f_2_1_0 - f_0_1_2
       + f_2_2_1 - f_0_0_1
       + f_2_0_1 - f_0_2_1)
    v = (f_1_2_1 - f_1_0_1
       + f_1_2_2 - f_1_0_0
       + f_1_2_0 - f_1_0_2
       + f_2_2_1 - f_0_0_1
       - f_2_0_1 + f_0_2_1)
    w = (f_1_1_2 - f_1_1_0
       + f_1_2_2 - f_1_0_0
       - f_1_2_0 + f_1_0_2
       + f_2_1_2 - f_0_1_0
       - f_2_1_0 + f_0_1_2)
    res_p = 1.0 / p
    u = u * res_p
    v = v * res_p
    w = w * res_p
    uxu = u * u + v * v + w * w

    # compute e dot u
    exu_1_1_1 = 0
    exu_2_1_1 = u
    exu_0_1_1 = -u
    exu_1_2_1 = v
    exu_1_0_1 = -v
    exu_1_1_2 = w
    exu_1_1_0 = -w
    exu_1_2_2 = v + w
    exu_1_0_0 = -v - w
    exu_1_2_0 = v - w
    exu_1_0_2 = -v + w
    exu_2_1_2 = u + w
    exu_0_1_0 = -u - w
    exu_2_1_0 = u - w
    exu_0_1_2 = -u + w
    exu_2_2_1 = u + v
    exu_0_0_1 = -u - v
    exu_2_0_1 = u - v
    exu_0_2_1 = -u + v

    # compute equilibrium dist
    factor_1 = 1.5
    factor_2 = 4.5
    weight_0 = 0.33333333
    weight_1 = 0.05555555
    weight_2 = 0.02777777
    f_eq_1_1_1 = weight_0 * (p * (factor_1 * (- uxu) + 1.0))
    f_eq_2_1_1 = weight_1 * (p * (factor_1 * (2.0 * exu_2_1_1 - uxu) + factor_2 * (exu_2_1_1 * exu_2_1_1) + 1.0))
    f_eq_0_1_1 = weight_1 * (p * (factor_1 * (2.0 * exu_0_1_1 - uxu) + factor_2 * (exu_0_1_1 * exu_0_1_1) + 1.0))
    f_eq_1_2_1 = weight_1 * (p * (factor_1 * (2.0 * exu_1_2_1 - uxu) + factor_2 * (exu_1_2_1 * exu_1_2_1) + 1.0))
    f_eq_1_0_1 = weight_1 * (p * (factor_1 * (2.0 * exu_1_0_1 - uxu) + factor_2 * (exu_1_2_1 * exu_1_2_1) + 1.0))
    f_eq_1_1_2 = weight_1 * (p * (factor_1 * (2.0 * exu_1_1_2 - uxu) + factor_2 * (exu_1_1_2 * exu_1_1_2) + 1.0))
    f_eq_1_1_0 = weight_1 * (p * (factor_1 * (2.0 * exu_1_1_0 - uxu) + factor_2 * (exu_1_1_0 * exu_1_1_0) + 1.0))
    f_eq_1_2_2 = weight_2 * (p * (factor_1 * (2.0 * exu_1_2_2 - uxu) + factor_2 * (exu_1_2_2 * exu_1_2_2) + 1.0))
    f_eq_1_0_0 = weight_2 * (p * (factor_1 * (2.0 * exu_1_0_0 - uxu) + factor_2 * (exu_1_0_0 * exu_1_0_0) + 1.0))
    f_eq_1_2_0 = weight_2 * (p * (factor_1 * (2.0 * exu_1_2_0 - uxu) + factor_2 * (exu_1_2_0 * exu_1_2_0) + 1.0))
    f_eq_1_0_2 = weight_2 * (p * (factor_1 * (2.0 * exu_1_0_2 - uxu) + factor_2 * (exu_1_0_2 * exu_1_0_2) + 1.0))
    f_eq_2_1_2 = weight_2 * (p * (factor_1 * (2.0 * exu_2_1_2 - uxu) + factor_2 * (exu_2_1_2 * exu_2_1_2) + 1.0))
    f_eq_0_1_0 = weight_2 * (p * (factor_1 * (2.0 * exu_0_1_0 - uxu) + factor_2 * (exu_0_1_0 * exu_0_1_0) + 1.0))
    f_eq_2_1_0 = weight_2 * (p * (factor_1 * (2.0 * exu_2_1_0 - uxu) + factor_2 * (exu_2_1_0 * exu_2_1_0) + 1.0))
    f_eq_0_1_2 = weight_2 * (p * (factor_1 * (2.0 * exu_0_1_2 - uxu) + factor_2 * (exu_0_1_2 * exu_0_1_2) + 1.0))
    f_eq_2_2_1 = weight_2 * (p * (factor_1 * (2.0 * exu_2_2_1 - uxu) + factor_2 * (exu_2_2_1 * exu_2_2_1) + 1.0))
    f_eq_0_0_1 = weight_2 * (p * (factor_1 * (2.0 * exu_0_0_1 - uxu) + factor_2 * (exu_0_0_1 * exu_0_0_1) + 1.0))
    f_eq_2_0_1 = weight_2 * (p * (factor_1 * (2.0 * exu_2_0_1 - uxu) + factor_2 * (exu_2_0_1 * exu_2_0_1) + 1.0))
    f_eq_0_2_1 = weight_2 * (p * (factor_1 * (2.0 * exu_0_2_1 - uxu) + factor_2 * (exu_0_2_1 * exu_0_2_1) + 1.0))

    # set next lattice state
    inv_tau = (1.0 / tau)
    warp_set_f(f1, f_1_1_1 - inv_tau * (f_1_1_1 - f_eq_1_1_1), 0, x, y, z, width, height, length)
    warp_set_f(f1, f_2_1_1 - inv_tau * (f_2_1_1 - f_eq_2_1_1), 1, x + 1, y, z, width, height, length)
    warp_set_f(f1, f_0_1_1 - inv_tau * (f_0_1_1 - f_eq_0_1_1), 2, x - 1, y, z, width, height, length)
    warp_set_f(f1, f_1_2_1 - inv_tau * (f_1_2_1 - f_eq_1_2_1), 3, x, y + 1, z, width, height, length)
    warp_set_f(f1, f_1_0_1 - inv_tau * (f_1_0_1 - f_eq_1_0_1), 4, x, y - 1, z, width, height, length)
    warp_set_f(f1, f_1_1_2 - inv_tau * (f_1_1_2 - f_eq_1_1_2), 5, x, y, z + 1, width, height, length)
    warp_set_f(f1, f_1_1_0 - inv_tau * (f_1_1_0 - f_eq_1_1_0), 6, x, y, z - 1, width, height, length)
    warp_set_f(f1, f_1_2_2 - inv_tau * (f_1_2_2 - f_eq_1_2_2), 7, x, y + 1, z + 1, width, height, length)
    warp_set_f(f1, f_1_0_0 - inv_tau * (f_1_0_0 - f_eq_1_0_0), 8, x, y - 1, z - 1, width, height, length)
    warp_set_f(f1, f_1_2_0 - inv_tau * (f_1_2_0 - f_eq_1_2_0), 9, x, y + 1, z - 1, width, height, length)
    warp_set_f(f1, f_1_0_2 - inv_tau * (f_1_0_2 - f_eq_1_0_2), 10, x, y - 1, z + 1, width, height, length)
    warp_set_f(f1, f_2_1_2 - inv_tau * (f_2_1_2 - f_eq_2_1_2), 11, x + 1, y, z + 1, width, height, length)
    warp_set_f(f1, f_0_1_0 - inv_tau * (f_0_1_0 - f_eq_0_1_0), 12, x - 1, y, z - 1, width, height, length)
    warp_set_f(f1, f_2_1_0 - inv_tau * (f_2_1_0 - f_eq_2_1_0), 13, x + 1, y, z - 1, width, height, length)
    warp_set_f(f1, f_0_1_2 - inv_tau * (f_0_1_2 - f_eq_0_1_2), 14, x - 1, y, z + 1, width, height, length)
    warp_set_f(f1, f_2_2_1 - inv_tau * (f_2_2_1 - f_eq_2_2_1), 15, x + 1, y + 1, z, width, height, length)
    warp_set_f(f1, f_0_0_1 - inv_tau * (f_0_0_1 - f_eq_0_0_1), 16, x - 1, y - 1, z, width, height, length)
    warp_set_f(f1, f_2_0_1 - inv_tau * (f_2_0_1 - f_eq_2_0_1), 17, x + 1, y - 1, z, width, height, length)
    warp_set_f(f1, f_0_2_1 - inv_tau * (f_0_2_1 - f_eq_0_2_1), 18, x - 1, y + 1, z, width, height, length)

@wp.kernel
def warp_initialize_taylor_green(
    f: wp.array4d(dtype=wp.float32),
    dx: float,
    vel: float,
    start_x: int,
    start_y: int,
    start_z: int,
):

    # get index
    i, j, k = wp.tid()

    # get real pos
    x = wp.float(i + start_x) * dx
    y = wp.float(j + start_y) * dx
    z = wp.float(k + start_z) * dx

    # compute u
    u = vel * wp.sin(x) * wp.cos(y) * wp.cos(z)
    v = -vel * wp.cos(x) * wp.sin(y) * wp.cos(z)
    w = 0.0

    # compute p
    p = (
        3.0
        * vel
        * vel
        * (1.0 / 16.0)
        * (wp.cos(2.0 * x) + wp.cos(2.0 * y) * (wp.cos(2.0 * z) + 2.0))
        + 1.0
    )

    # compute u X u
    uxu = u * u + v * v + w * w

    # compute e dot u
    exu_1_1_1 = 0.0
    exu_2_1_1 = u
    exu_0_1_1 = -u
    exu_1_2_1 = v
    exu_1_0_1 = -v
    exu_1_1_2 = w
    exu_1_1_0 = -w
    exu_1_2_2 = v + w
    exu_1_0_0 = -v - w
    exu_1_2_0 = v - w
    exu_1_0_2 = -v + w
    exu_2_1_2 = u + w
    exu_0_1_0 = -u - w
    exu_2_1_0 = u - w
    exu_0_1_2 = -u + w
    exu_2_2_1 = u + v
    exu_0_0_1 = -u - v
    exu_2_0_1 = u - v
    exu_0_2_1 = -u + v

    # compute equilibrium dist
    factor_1 = 1.5
    factor_2 = 4.5
    weight_0 = 0.33333333
    weight_1 = 0.05555555
    weight_2 = 0.02777777
    f_eq_1_1_1 = weight_0 * (p * (factor_1 * (-uxu) + 1.0))
    f_eq_2_1_1 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_2_1_1 - uxu)
            + factor_2 * (exu_2_1_1 * exu_2_1_1)
            + 1.0
        )
    )
    f_eq_0_1_1 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_0_1_1 - uxu)
            + factor_2 * (exu_0_1_1 * exu_0_1_1)
            + 1.0
        )
    )
    f_eq_1_2_1 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_1_2_1 - uxu)
            + factor_2 * (exu_1_2_1 * exu_1_2_1)
            + 1.0
        )
    )
    f_eq_1_0_1 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_1_0_1 - uxu)
            + factor_2 * (exu_1_2_1 * exu_1_2_1)
            + 1.0
        )
    )
    f_eq_1_1_2 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_1_1_2 - uxu)
            + factor_2 * (exu_1_1_2 * exu_1_1_2)
            + 1.0
        )
    )
    f_eq_1_1_0 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_1_1_0 - uxu)
            + factor_2 * (exu_1_1_0 * exu_1_1_0)
            + 1.0
        )
    )
    f_eq_1_2_2 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_1_2_2 - uxu)
            + factor_2 * (exu_1_2_2 * exu_1_2_2)
            + 1.0
        )
    )
    f_eq_1_0_0 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_1_0_0 - uxu)
            + factor_2 * (exu_1_0_0 * exu_1_0_0)
            + 1.0
        )
    )
    f_eq_1_2_0 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_1_2_0 - uxu)
            + factor_2 * (exu_1_2_0 * exu_1_2_0)
            + 1.0
        )
    )
    f_eq_1_0_2 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_1_0_2 - uxu)
            + factor_2 * (exu_1_0_2 * exu_1_0_2)
            + 1.0
        )
    )
    f_eq_2_1_2 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_2_1_2 - uxu)
            + factor_2 * (exu_2_1_2 * exu_2_1_2)
            + 1.0
        )
    )
    f_eq_0_1_0 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_0_1_0 - uxu)
            + factor_2 * (exu_0_1_0 * exu_0_1_0)
            + 1.0
        )
    )
    f_eq_2_1_0 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_2_1_0 - uxu)
            + factor_2 * (exu_2_1_0 * exu_2_1_0)
            + 1.0
        )
    )
    f_eq_0_1_2 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_0_1_2 - uxu)
            + factor_2 * (exu_0_1_2 * exu_0_1_2)
            + 1.0
        )
    )
    f_eq_2_2_1 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_2_2_1 - uxu)
            + factor_2 * (exu_2_2_1 * exu_2_2_1)
            + 1.0
        )
    )
    f_eq_0_0_1 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_0_0_1 - uxu)
            + factor_2 * (exu_0_0_1 * exu_0_0_1)
            + 1.0
        )
    )
    f_eq_2_0_1 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_2_0_1 - uxu)
            + factor_2 * (exu_2_0_1 * exu_2_0_1)
            + 1.0
        )
    )
    f_eq_0_2_1 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_0_2_1 - uxu)
            + factor_2 * (exu_0_2_1 * exu_0_2_1)
            + 1.0
        )
    )

    # set next lattice state
    f[0, i, j, k] = f_eq_1_1_1
    f[1, i, j, k] = f_eq_2_1_1
    f[2, i, j, k] = f_eq_0_1_1
    f[3, i, j, k] = f_eq_1_2_1
    f[4, i, j, k] = f_eq_1_0_1
    f[5, i, j, k] = f_eq_1_1_2
    f[6, i, j, k] = f_eq_1_1_0
    f[7, i, j, k] = f_eq_1_2_2
    f[8, i, j, k] = f_eq_1_0_0
    f[9, i, j, k] = f_eq_1_2_0
    f[10, i, j, k] = f_eq_1_0_2
    f[11, i, j, k] = f_eq_2_1_2
    f[12, i, j, k] = f_eq_0_1_0
    f[13, i, j, k] = f_eq_2_1_0
    f[14, i, j, k] = f_eq_0_1_2
    f[15, i, j, k] = f_eq_2_2_1
    f[16, i, j, k] = f_eq_0_0_1
    f[17, i, j, k] = f_eq_2_0_1
    f[18, i, j, k] = f_eq_0_2_1


def warp_initialize_f(f, dx: float):
    # Get inputs
    cs = 1.0 / np.sqrt(3.0)
    vel = 0.1 * cs

    # Launch kernel
    wp.launch(
        kernel=warp_initialize_taylor_green,
        dim=list(f.shape[1:]),
        inputs=[f, dx, vel, 0, 0, 0],
        device=f.device,
    )

    return f


def warp_apply_collide_stream(f0, f1, tau: float):
    # Apply streaming and collision step
    wp.launch(
        kernel=warp_collide_stream,
        dim=list(f0.shape[1:]),
        inputs=[f0, f1, f0.shape[1], f0.shape[2], f0.shape[3], tau],
        device=f0.device,
    )

    return f1, f0


@cuda.jit("void(float32[:,:,:,::1], float32, int32, int32, int32, int32, int32, int32, int32)", device=True)
def numba_set_f(
    f: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    value: float,
    q: int,
    i: int,
    j: int,
    k: int,
    width: int,
    height: int,
    length: int,
):
    # Modulo
    if i < 0:
        i += width
    if j < 0:
        j += height
    if k < 0:
        k += length
    if i >= width:
        i -= width
    if j >= height:
        j -= height
    if k >= length:
        k -= length
    f[i, j, k, q] = value

#@cuda.jit
@cuda.jit("void(float32[:,:,:,::1], float32[:,:,:,::1], int32, int32, int32, float32)")
def numba_collide_stream(
    f0: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    f1: numba.cuda.cudadrv.devicearray.DeviceNDArray,
    width: int,
    height: int,
    length: int,
    tau: float,
):

    x, y, z = cuda.grid(3)

    # sample needed points
    f_1_1_1 = f0[x, y, z, 0]
    f_2_1_1 = f0[x, y, z, 1]
    f_0_1_1 = f0[x, y, z, 2]
    f_1_2_1 = f0[x, y, z, 3]
    f_1_0_1 = f0[x, y, z, 4]
    f_1_1_2 = f0[x, y, z, 5]
    f_1_1_0 = f0[x, y, z, 6]
    f_1_2_2 = f0[x, y, z, 7]
    f_1_0_0 = f0[x, y, z, 8]
    f_1_2_0 = f0[x, y, z, 9]
    f_1_0_2 = f0[x, y, z, 10]
    f_2_1_2 = f0[x, y, z, 11]
    f_0_1_0 = f0[x, y, z, 12]
    f_2_1_0 = f0[x, y, z, 13]
    f_0_1_2 = f0[x, y, z, 14]
    f_2_2_1 = f0[x, y, z, 15]
    f_0_0_1 = f0[x, y, z, 16]
    f_2_0_1 = f0[x, y, z, 17]
    f_0_2_1 = f0[x, y, z, 18]

    # compute u and p
    p = (f_1_1_1
       + f_2_1_1 + f_0_1_1
       + f_1_2_1 + f_1_0_1
       + f_1_1_2 + f_1_1_0
       + f_1_2_2 + f_1_0_0
       + f_1_2_0 + f_1_0_2
       + f_2_1_2 + f_0_1_0
       + f_2_1_0 + f_0_1_2
       + f_2_2_1 + f_0_0_1
       + f_2_0_1 + f_0_2_1)
    u = (f_2_1_1 - f_0_1_1
       + f_2_1_2 - f_0_1_0
       + f_2_1_0 - f_0_1_2
       + f_2_2_1 - f_0_0_1
       + f_2_0_1 - f_0_2_1)
    v = (f_1_2_1 - f_1_0_1
       + f_1_2_2 - f_1_0_0
       + f_1_2_0 - f_1_0_2
       + f_2_2_1 - f_0_0_1
       - f_2_0_1 + f_0_2_1)
    w = (f_1_1_2 - f_1_1_0
       + f_1_2_2 - f_1_0_0
       - f_1_2_0 + f_1_0_2
       + f_2_1_2 - f_0_1_0
       - f_2_1_0 + f_0_1_2)
    res_p = numba.float32(1.0) / p
    u = u * res_p
    v = v * res_p
    w = w * res_p
    uxu = u * u + v * v + w * w

    # compute e dot u
    exu_1_1_1 = numba.float32(0.0)
    exu_2_1_1 = u
    exu_0_1_1 = -u
    exu_1_2_1 = v
    exu_1_0_1 = -v
    exu_1_1_2 = w
    exu_1_1_0 = -w
    exu_1_2_2 = v + w
    exu_1_0_0 = -v - w
    exu_1_2_0 = v - w
    exu_1_0_2 = -v + w
    exu_2_1_2 = u + w
    exu_0_1_0 = -u - w
    exu_2_1_0 = u - w
    exu_0_1_2 = -u + w
    exu_2_2_1 = u + v
    exu_0_0_1 = -u - v
    exu_2_0_1 = u - v
    exu_0_2_1 = -u + v

    # compute equilibrium dist
    factor_1 = numba.float32(1.5)
    factor_2 = numba.float32(4.5)
    weight_0 = numba.float32(0.33333333)
    weight_1 = numba.float32(0.05555555)
    weight_2 = numba.float32(0.02777777)
 
    f_eq_1_1_1 = weight_0 * (p * (factor_1 * (- uxu) + numba.float32(1.0)))
    f_eq_2_1_1 = weight_1 * (p * (factor_1 * (numba.float32(2.0) * exu_2_1_1 - uxu) + factor_2 * (exu_2_1_1 * exu_2_1_1) + numba.float32(1.0)))
    f_eq_0_1_1 = weight_1 * (p * (factor_1 * (numba.float32(2.0) * exu_0_1_1 - uxu) + factor_2 * (exu_0_1_1 * exu_0_1_1) + numba.float32(1.0)))
    f_eq_1_2_1 = weight_1 * (p * (factor_1 * (numba.float32(2.0) * exu_1_2_1 - uxu) + factor_2 * (exu_1_2_1 * exu_1_2_1) + numba.float32(1.0)))
    f_eq_1_0_1 = weight_1 * (p * (factor_1 * (numba.float32(2.0) * exu_1_0_1 - uxu) + factor_2 * (exu_1_2_1 * exu_1_2_1) + numba.float32(1.0)))
    f_eq_1_1_2 = weight_1 * (p * (factor_1 * (numba.float32(2.0) * exu_1_1_2 - uxu) + factor_2 * (exu_1_1_2 * exu_1_1_2) + numba.float32(1.0)))
    f_eq_1_1_0 = weight_1 * (p * (factor_1 * (numba.float32(2.0) * exu_1_1_0 - uxu) + factor_2 * (exu_1_1_0 * exu_1_1_0) + numba.float32(1.0)))
    f_eq_1_2_2 = weight_2 * (p * (factor_1 * (numba.float32(2.0) * exu_1_2_2 - uxu) + factor_2 * (exu_1_2_2 * exu_1_2_2) + numba.float32(1.0)))
    f_eq_1_0_0 = weight_2 * (p * (factor_1 * (numba.float32(2.0) * exu_1_0_0 - uxu) + factor_2 * (exu_1_0_0 * exu_1_0_0) + numba.float32(1.0)))
    f_eq_1_2_0 = weight_2 * (p * (factor_1 * (numba.float32(2.0) * exu_1_2_0 - uxu) + factor_2 * (exu_1_2_0 * exu_1_2_0) + numba.float32(1.0)))
    f_eq_1_0_2 = weight_2 * (p * (factor_1 * (numba.float32(2.0) * exu_1_0_2 - uxu) + factor_2 * (exu_1_0_2 * exu_1_0_2) + numba.float32(1.0)))
    f_eq_2_1_2 = weight_2 * (p * (factor_1 * (numba.float32(2.0) * exu_2_1_2 - uxu) + factor_2 * (exu_2_1_2 * exu_2_1_2) + numba.float32(1.0)))
    f_eq_0_1_0 = weight_2 * (p * (factor_1 * (numba.float32(2.0) * exu_0_1_0 - uxu) + factor_2 * (exu_0_1_0 * exu_0_1_0) + numba.float32(1.0)))
    f_eq_2_1_0 = weight_2 * (p * (factor_1 * (numba.float32(2.0) * exu_2_1_0 - uxu) + factor_2 * (exu_2_1_0 * exu_2_1_0) + numba.float32(1.0)))
    f_eq_0_1_2 = weight_2 * (p * (factor_1 * (numba.float32(2.0) * exu_0_1_2 - uxu) + factor_2 * (exu_0_1_2 * exu_0_1_2) + numba.float32(1.0)))
    f_eq_2_2_1 = weight_2 * (p * (factor_1 * (numba.float32(2.0) * exu_2_2_1 - uxu) + factor_2 * (exu_2_2_1 * exu_2_2_1) + numba.float32(1.0)))
    f_eq_0_0_1 = weight_2 * (p * (factor_1 * (numba.float32(2.0) * exu_0_0_1 - uxu) + factor_2 * (exu_0_0_1 * exu_0_0_1) + numba.float32(1.0)))
    f_eq_2_0_1 = weight_2 * (p * (factor_1 * (numba.float32(2.0) * exu_2_0_1 - uxu) + factor_2 * (exu_2_0_1 * exu_2_0_1) + numba.float32(1.0)))
    f_eq_0_2_1 = weight_2 * (p * (factor_1 * (numba.float32(2.0) * exu_0_2_1 - uxu) + factor_2 * (exu_0_2_1 * exu_0_2_1) + numba.float32(1.0)))

    # set next lattice state
    inv_tau = numba.float32((numba.float32(1.0) / tau))
    numba_set_f(f1, f_1_1_1 - inv_tau * (f_1_1_1 - f_eq_1_1_1), 0, x, y, z, width, height, length)
    numba_set_f(f1, f_2_1_1 - inv_tau * (f_2_1_1 - f_eq_2_1_1), 1, x + 1, y, z, width, height, length)
    numba_set_f(f1, f_0_1_1 - inv_tau * (f_0_1_1 - f_eq_0_1_1), 2, x - 1, y, z, width, height, length)
    numba_set_f(f1, f_1_2_1 - inv_tau * (f_1_2_1 - f_eq_1_2_1), 3, x, y + 1, z, width, height, length)
    numba_set_f(f1, f_1_0_1 - inv_tau * (f_1_0_1 - f_eq_1_0_1), 4, x, y - 1, z, width, height, length)
    numba_set_f(f1, f_1_1_2 - inv_tau * (f_1_1_2 - f_eq_1_1_2), 5, x, y, z + 1, width, height, length)
    numba_set_f(f1, f_1_1_0 - inv_tau * (f_1_1_0 - f_eq_1_1_0), 6, x, y, z - 1, width, height, length)
    numba_set_f(f1, f_1_2_2 - inv_tau * (f_1_2_2 - f_eq_1_2_2), 7, x, y + 1, z + 1, width, height, length)
    numba_set_f(f1, f_1_0_0 - inv_tau * (f_1_0_0 - f_eq_1_0_0), 8, x, y - 1, z - 1, width, height, length)
    numba_set_f(f1, f_1_2_0 - inv_tau * (f_1_2_0 - f_eq_1_2_0), 9, x, y + 1, z - 1, width, height, length)
    numba_set_f(f1, f_1_0_2 - inv_tau * (f_1_0_2 - f_eq_1_0_2), 10, x, y - 1, z + 1, width, height, length)
    numba_set_f(f1, f_2_1_2 - inv_tau * (f_2_1_2 - f_eq_2_1_2), 11, x + 1, y, z + 1, width, height, length)
    numba_set_f(f1, f_0_1_0 - inv_tau * (f_0_1_0 - f_eq_0_1_0), 12, x - 1, y, z - 1, width, height, length)
    numba_set_f(f1, f_2_1_0 - inv_tau * (f_2_1_0 - f_eq_2_1_0), 13, x + 1, y, z - 1, width, height, length)
    numba_set_f(f1, f_0_1_2 - inv_tau * (f_0_1_2 - f_eq_0_1_2), 14, x - 1, y, z + 1, width, height, length)
    numba_set_f(f1, f_2_2_1 - inv_tau * (f_2_2_1 - f_eq_2_2_1), 15, x + 1, y + 1, z, width, height, length)
    numba_set_f(f1, f_0_0_1 - inv_tau * (f_0_0_1 - f_eq_0_0_1), 16, x - 1, y - 1, z, width, height, length)
    numba_set_f(f1, f_2_0_1 - inv_tau * (f_2_0_1 - f_eq_2_0_1), 17, x + 1, y - 1, z, width, height, length)
    numba_set_f(f1, f_0_2_1 - inv_tau * (f_0_2_1 - f_eq_0_2_1), 18, x - 1, y + 1, z, width, height, length)


@cuda.jit
def numba_initialize_taylor_green(
    f,
    dx,
    vel,
    start_x,
    start_y,
    start_z,
):

    i, j, k = cuda.grid(3)

    # get real pos
    x = numba.float32(i + start_x) * dx
    y = numba.float32(j + start_y) * dx
    z = numba.float32(k + start_z) * dx

    # compute u
    u = vel * math.sin(x) * math.cos(y) * math.cos(z)
    v = -vel * math.cos(x) * math.sin(y) * math.cos(z)
    w = 0.0

    # compute p
    p = (
        3.0
        * vel
        * vel
        * (1.0 / 16.0)
        * (math.cos(2.0 * x) + math.cos(2.0 * y) * (math.cos(2.0 * z) + 2.0))
        + 1.0
    )

    # compute u X u
    uxu = u * u + v * v + w * w

    # compute e dot u
    exu_1_1_1 = 0.0
    exu_2_1_1 = u
    exu_0_1_1 = -u
    exu_1_2_1 = v
    exu_1_0_1 = -v
    exu_1_1_2 = w
    exu_1_1_0 = -w
    exu_1_2_2 = v + w
    exu_1_0_0 = -v - w
    exu_1_2_0 = v - w
    exu_1_0_2 = -v + w
    exu_2_1_2 = u + w
    exu_0_1_0 = -u - w
    exu_2_1_0 = u - w
    exu_0_1_2 = -u + w
    exu_2_2_1 = u + v
    exu_0_0_1 = -u - v
    exu_2_0_1 = u - v
    exu_0_2_1 = -u + v

    # compute equilibrium dist
    factor_1 = 1.5
    factor_2 = 4.5
    weight_0 = 0.33333333
    weight_1 = 0.05555555
    weight_2 = 0.02777777
    f_eq_1_1_1 = weight_0 * (p * (factor_1 * (-uxu) + 1.0))
    f_eq_2_1_1 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_2_1_1 - uxu)
            + factor_2 * (exu_2_1_1 * exu_2_1_1)
            + 1.0
        )
    )
    f_eq_0_1_1 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_0_1_1 - uxu)
            + factor_2 * (exu_0_1_1 * exu_0_1_1)
            + 1.0
        )
    )
    f_eq_1_2_1 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_1_2_1 - uxu)
            + factor_2 * (exu_1_2_1 * exu_1_2_1)
            + 1.0
        )
    )
    f_eq_1_0_1 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_1_0_1 - uxu)
            + factor_2 * (exu_1_2_1 * exu_1_2_1)
            + 1.0
        )
    )
    f_eq_1_1_2 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_1_1_2 - uxu)
            + factor_2 * (exu_1_1_2 * exu_1_1_2)
            + 1.0
        )
    )
    f_eq_1_1_0 = weight_1 * (
        p
        * (
            factor_1 * (2.0 * exu_1_1_0 - uxu)
            + factor_2 * (exu_1_1_0 * exu_1_1_0)
            + 1.0
        )
    )
    f_eq_1_2_2 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_1_2_2 - uxu)
            + factor_2 * (exu_1_2_2 * exu_1_2_2)
            + 1.0
        )
    )
    f_eq_1_0_0 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_1_0_0 - uxu)
            + factor_2 * (exu_1_0_0 * exu_1_0_0)
            + 1.0
        )
    )
    f_eq_1_2_0 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_1_2_0 - uxu)
            + factor_2 * (exu_1_2_0 * exu_1_2_0)
            + 1.0
        )
    )
    f_eq_1_0_2 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_1_0_2 - uxu)
            + factor_2 * (exu_1_0_2 * exu_1_0_2)
            + 1.0
        )
    )
    f_eq_2_1_2 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_2_1_2 - uxu)
            + factor_2 * (exu_2_1_2 * exu_2_1_2)
            + 1.0
        )
    )
    f_eq_0_1_0 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_0_1_0 - uxu)
            + factor_2 * (exu_0_1_0 * exu_0_1_0)
            + 1.0
        )
    )
    f_eq_2_1_0 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_2_1_0 - uxu)
            + factor_2 * (exu_2_1_0 * exu_2_1_0)
            + 1.0
        )
    )
    f_eq_0_1_2 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_0_1_2 - uxu)
            + factor_2 * (exu_0_1_2 * exu_0_1_2)
            + 1.0
        )
    )
    f_eq_2_2_1 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_2_2_1 - uxu)
            + factor_2 * (exu_2_2_1 * exu_2_2_1)
            + 1.0
        )
    )
    f_eq_0_0_1 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_0_0_1 - uxu)
            + factor_2 * (exu_0_0_1 * exu_0_0_1)
            + 1.0
        )
    )
    f_eq_2_0_1 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_2_0_1 - uxu)
            + factor_2 * (exu_2_0_1 * exu_2_0_1)
            + 1.0
        )
    )
    f_eq_0_2_1 = weight_2 * (
        p
        * (
            factor_1 * (2.0 * exu_0_2_1 - uxu)
            + factor_2 * (exu_0_2_1 * exu_0_2_1)
            + 1.0
        )
    )

    # set next lattice state
    f[i, j, k, 0] = f_eq_1_1_1
    f[i, j, k, 1] = f_eq_2_1_1
    f[i, j, k, 2] = f_eq_0_1_1
    f[i, j, k, 3] = f_eq_1_2_1
    f[i, j, k, 4] = f_eq_1_0_1
    f[i, j, k, 5] = f_eq_1_1_2
    f[i, j, k, 6] = f_eq_1_1_0
    f[i, j, k, 7] = f_eq_1_2_2
    f[i, j, k, 8] = f_eq_1_0_0
    f[i, j, k, 9] = f_eq_1_2_0
    f[ i, j, k, 10] = f_eq_1_0_2
    f[ i, j, k, 11] = f_eq_2_1_2
    f[ i, j, k, 12] = f_eq_0_1_0
    f[ i, j, k, 13] = f_eq_2_1_0
    f[ i, j, k, 14] = f_eq_0_1_2
    f[ i, j, k, 15] = f_eq_2_2_1
    f[ i, j, k, 16] = f_eq_0_0_1
    f[ i, j, k, 17] = f_eq_2_0_1
    f[ i, j, k, 18] = f_eq_0_2_1


def numba_initialize_f(f, dx: float):
    # Get inputs
    cs = 1.0 / np.sqrt(3.0)
    vel = 0.1 * cs

    # Launch kernel
    blockdim = (16, 16, 1)
    griddim = (
            int(np.ceil(f.shape[0] / blockdim[0])),
            int(np.ceil(f.shape[1] / blockdim[1])),
            int(np.ceil(f.shape[2] / blockdim[2])),
    )
    numba_initialize_taylor_green[griddim, blockdim](
        f, dx, vel, 0, 0, 0
    )

    return f

def numba_apply_collide_stream(f0, f1, tau: float):
    # Apply streaming and collision step
    blockdim = (8, 8, 8)
    griddim = (
            int(np.ceil(f0.shape[0] / blockdim[0])),
            int(np.ceil(f0.shape[1] / blockdim[1])),
            int(np.ceil(f0.shape[2] / blockdim[2])),
    )
    numba_collide_stream[griddim, blockdim](
        f0, f1, f0.shape[0], f0.shape[1], f0.shape[2], tau
    )

    return f1, f0

@partial(jit, static_argnums=(1), donate_argnums=(0))
def jax_apply_collide_stream(f, tau: float):

    # Get f directions
    f_1_1_1 = f[:, :, :, 0]
    f_2_1_1 = f[:, :, :, 1]
    f_0_1_1 = f[:, :, :, 2]
    f_1_2_1 = f[:, :, :, 3]
    f_1_0_1 = f[:, :, :, 4]
    f_1_1_2 = f[:, :, :, 5]
    f_1_1_0 = f[:, :, :, 6]
    f_1_2_2 = f[:, :, :, 7]
    f_1_0_0 = f[:, :, :, 8]
    f_1_2_0 = f[:, :, :, 9]
    f_1_0_2 = f[:, :, :, 10]
    f_2_1_2 = f[:, :, :, 11]
    f_0_1_0 = f[:, :, :, 12]
    f_2_1_0 = f[:, :, :, 13]
    f_0_1_2 = f[:, :, :, 14]
    f_2_2_1 = f[:, :, :, 15]
    f_0_0_1 = f[:, :, :, 16]
    f_2_0_1 = f[:, :, :, 17]
    f_0_2_1 = f[:, :, :, 18]

    # compute u and p
    p = (f_1_1_1
       + f_2_1_1 + f_0_1_1
       + f_1_2_1 + f_1_0_1
       + f_1_1_2 + f_1_1_0
       + f_1_2_2 + f_1_0_0
       + f_1_2_0 + f_1_0_2
       + f_2_1_2 + f_0_1_0
       + f_2_1_0 + f_0_1_2
       + f_2_2_1 + f_0_0_1
       + f_2_0_1 + f_0_2_1)
    u = (f_2_1_1 - f_0_1_1
       + f_2_1_2 - f_0_1_0
       + f_2_1_0 - f_0_1_2
       + f_2_2_1 - f_0_0_1
       + f_2_0_1 - f_0_2_1)
    v = (f_1_2_1 - f_1_0_1
       + f_1_2_2 - f_1_0_0
       + f_1_2_0 - f_1_0_2
       + f_2_2_1 - f_0_0_1
       - f_2_0_1 + f_0_2_1)
    w = (f_1_1_2 - f_1_1_0
       + f_1_2_2 - f_1_0_0
       - f_1_2_0 + f_1_0_2
       + f_2_1_2 - f_0_1_0
       - f_2_1_0 + f_0_1_2)
    res_p = 1.0 / p
    u = u * res_p
    v = v * res_p
    w = w * res_p
    uxu = u * u + v * v + w * w

    # compute e dot u
    exu_1_1_1 = 0
    exu_2_1_1 = u
    exu_0_1_1 = -u
    exu_1_2_1 = v
    exu_1_0_1 = -v
    exu_1_1_2 = w
    exu_1_1_0 = -w
    exu_1_2_2 = v + w
    exu_1_0_0 = -v - w
    exu_1_2_0 = v - w
    exu_1_0_2 = -v + w
    exu_2_1_2 = u + w
    exu_0_1_0 = -u - w
    exu_2_1_0 = u - w
    exu_0_1_2 = -u + w
    exu_2_2_1 = u + v
    exu_0_0_1 = -u - v
    exu_2_0_1 = u - v
    exu_0_2_1 = -u + v

    # compute equilibrium dist
    factor_1 = 1.5
    factor_2 = 4.5
    weight_0 = 0.33333333
    weight_1 = 0.05555555
    weight_2 = 0.02777777
    f_eq_1_1_1 = weight_0 * (p * (factor_1 * (- uxu) + 1.0))
    f_eq_2_1_1 = weight_1 * (p * (factor_1 * (2.0 * exu_2_1_1 - uxu) + factor_2 * (exu_2_1_1 * exu_2_1_1) + 1.0))
    f_eq_0_1_1 = weight_1 * (p * (factor_1 * (2.0 * exu_0_1_1 - uxu) + factor_2 * (exu_0_1_1 * exu_0_1_1) + 1.0))
    f_eq_1_2_1 = weight_1 * (p * (factor_1 * (2.0 * exu_1_2_1 - uxu) + factor_2 * (exu_1_2_1 * exu_1_2_1) + 1.0))
    f_eq_1_0_1 = weight_1 * (p * (factor_1 * (2.0 * exu_1_0_1 - uxu) + factor_2 * (exu_1_2_1 * exu_1_2_1) + 1.0))
    f_eq_1_1_2 = weight_1 * (p * (factor_1 * (2.0 * exu_1_1_2 - uxu) + factor_2 * (exu_1_1_2 * exu_1_1_2) + 1.0))
    f_eq_1_1_0 = weight_1 * (p * (factor_1 * (2.0 * exu_1_1_0 - uxu) + factor_2 * (exu_1_1_0 * exu_1_1_0) + 1.0))
    f_eq_1_2_2 = weight_2 * (p * (factor_1 * (2.0 * exu_1_2_2 - uxu) + factor_2 * (exu_1_2_2 * exu_1_2_2) + 1.0))
    f_eq_1_0_0 = weight_2 * (p * (factor_1 * (2.0 * exu_1_0_0 - uxu) + factor_2 * (exu_1_0_0 * exu_1_0_0) + 1.0))
    f_eq_1_2_0 = weight_2 * (p * (factor_1 * (2.0 * exu_1_2_0 - uxu) + factor_2 * (exu_1_2_0 * exu_1_2_0) + 1.0))
    f_eq_1_0_2 = weight_2 * (p * (factor_1 * (2.0 * exu_1_0_2 - uxu) + factor_2 * (exu_1_0_2 * exu_1_0_2) + 1.0))
    f_eq_2_1_2 = weight_2 * (p * (factor_1 * (2.0 * exu_2_1_2 - uxu) + factor_2 * (exu_2_1_2 * exu_2_1_2) + 1.0))
    f_eq_0_1_0 = weight_2 * (p * (factor_1 * (2.0 * exu_0_1_0 - uxu) + factor_2 * (exu_0_1_0 * exu_0_1_0) + 1.0))
    f_eq_2_1_0 = weight_2 * (p * (factor_1 * (2.0 * exu_2_1_0 - uxu) + factor_2 * (exu_2_1_0 * exu_2_1_0) + 1.0))
    f_eq_0_1_2 = weight_2 * (p * (factor_1 * (2.0 * exu_0_1_2 - uxu) + factor_2 * (exu_0_1_2 * exu_0_1_2) + 1.0))
    f_eq_2_2_1 = weight_2 * (p * (factor_1 * (2.0 * exu_2_2_1 - uxu) + factor_2 * (exu_2_2_1 * exu_2_2_1) + 1.0))
    f_eq_0_0_1 = weight_2 * (p * (factor_1 * (2.0 * exu_0_0_1 - uxu) + factor_2 * (exu_0_0_1 * exu_0_0_1) + 1.0))
    f_eq_2_0_1 = weight_2 * (p * (factor_1 * (2.0 * exu_2_0_1 - uxu) + factor_2 * (exu_2_0_1 * exu_2_0_1) + 1.0))
    f_eq_0_2_1 = weight_2 * (p * (factor_1 * (2.0 * exu_0_2_1 - uxu) + factor_2 * (exu_0_2_1 * exu_0_2_1) + 1.0))

    # set next lattice state
    inv_tau = (1.0 / tau)
    f_1_1_1 = f_1_1_1 - inv_tau * (f_1_1_1 - f_eq_1_1_1)
    f_2_1_1 = f_2_1_1 - inv_tau * (f_2_1_1 - f_eq_2_1_1)
    f_0_1_1 = f_0_1_1 - inv_tau * (f_0_1_1 - f_eq_0_1_1)
    f_1_2_1 = f_1_2_1 - inv_tau * (f_1_2_1 - f_eq_1_2_1)
    f_1_0_1 = f_1_0_1 - inv_tau * (f_1_0_1 - f_eq_1_0_1)
    f_1_1_2 = f_1_1_2 - inv_tau * (f_1_1_2 - f_eq_1_1_2)
    f_1_1_0 = f_1_1_0 - inv_tau * (f_1_1_0 - f_eq_1_1_0)
    f_1_2_2 = f_1_2_2 - inv_tau * (f_1_2_2 - f_eq_1_2_2)
    f_1_0_0 = f_1_0_0 - inv_tau * (f_1_0_0 - f_eq_1_0_0)
    f_1_2_0 = f_1_2_0 - inv_tau * (f_1_2_0 - f_eq_1_2_0)
    f_1_0_2 = f_1_0_2 - inv_tau * (f_1_0_2 - f_eq_1_0_2)
    f_2_1_2 = f_2_1_2 - inv_tau * (f_2_1_2 - f_eq_2_1_2)
    f_0_1_0 = f_0_1_0 - inv_tau * (f_0_1_0 - f_eq_0_1_0)
    f_2_1_0 = f_2_1_0 - inv_tau * (f_2_1_0 - f_eq_2_1_0)
    f_0_1_2 = f_0_1_2 - inv_tau * (f_0_1_2 - f_eq_0_1_2)
    f_2_2_1 = f_2_2_1 - inv_tau * (f_2_2_1 - f_eq_2_2_1)
    f_0_0_1 = f_0_0_1 - inv_tau * (f_0_0_1 - f_eq_0_0_1)
    f_2_0_1 = f_2_0_1 - inv_tau * (f_2_0_1 - f_eq_2_0_1)
    f_0_2_1 = f_0_2_1 - inv_tau * (f_0_2_1 - f_eq_0_2_1)

    # Roll fs and concatenate
    f_2_1_1 = jnp.roll(f_2_1_1, -1, axis=0)
    f_0_1_1 = jnp.roll(f_0_1_1, 1, axis=0)
    f_1_2_1 = jnp.roll(f_1_2_1, -1, axis=1)
    f_1_0_1 = jnp.roll(f_1_0_1, 1, axis=1)
    f_1_1_2 = jnp.roll(f_1_1_2, -1, axis=2)
    f_1_1_0 = jnp.roll(f_1_1_0, 1, axis=2)
    f_1_2_2 = jnp.roll(jnp.roll(f_1_2_2, -1, axis=1), -1, axis=2)
    f_1_0_0 = jnp.roll(jnp.roll(f_1_0_0, 1, axis=1), 1, axis=2)
    f_1_2_0 = jnp.roll(jnp.roll(f_1_2_0, -1, axis=1), 1, axis=2)
    f_1_0_2 = jnp.roll(jnp.roll(f_1_0_2, 1, axis=1), -1, axis=2)
    f_2_1_2 = jnp.roll(jnp.roll(f_2_1_2, -1, axis=0), -1, axis=2)
    f_0_1_0 = jnp.roll(jnp.roll(f_0_1_0, 1, axis=0), 1, axis=2)
    f_2_1_0 = jnp.roll(jnp.roll(f_2_1_0, -1, axis=0), 1, axis=2)
    f_0_1_2 = jnp.roll(jnp.roll(f_0_1_2, 1, axis=0), -1, axis=2)
    f_2_2_1 = jnp.roll(jnp.roll(f_2_2_1, -1, axis=0), -1, axis=1)
    f_0_0_1 = jnp.roll(jnp.roll(f_0_0_1, 1, axis=0), 1, axis=1)
    f_2_0_1 = jnp.roll(jnp.roll(f_2_0_1, -1, axis=0), 1, axis=1)
    f_0_2_1 = jnp.roll(jnp.roll(f_0_2_1, 1, axis=0), -1, axis=1)

    return jnp.stack(
        [
            f_1_1_1,
            f_2_1_1,
            f_0_1_1,
            f_1_2_1,
            f_1_0_1,
            f_1_1_2,
            f_1_1_0,
            f_1_2_2,
            f_1_0_0,
            f_1_2_0,
            f_1_0_2,
            f_2_1_2,
            f_0_1_0,
            f_2_1_0,
            f_0_1_2,
            f_2_2_1,
            f_0_0_1,
            f_2_0_1,
            f_0_2_1,
        ],
        axis=-1,
    )



if __name__ == "__main__":

    # Sim Parameters
    n = 256
    tau = 0.505
    dx = 2.0 * np.pi / n
    nr_steps = 128

    # Bar plot
    backend = []
    mlups = []

    ######### Warp #########
    # Make f0, f1
    f0 = wp.empty((19, n, n, n), dtype=wp.float32, device="cuda:0")
    f1 = wp.empty((19, n, n, n), dtype=wp.float32, device="cuda:0")

    # Initialize f0
    f0 = warp_initialize_f(f0, dx)

    # Apply streaming and collision
    t0 = time.time()
    for _ in tqdm(range(nr_steps)):
        f0, f1 = warp_apply_collide_stream(f0, f1, tau)
    wp.synchronize()
    t1 = time.time()

    # Compute MLUPS
    mlups = (nr_steps * n * n * n) / (t1 - t0) / 1e6
    backend.append("Warp")
    print(mlups)
    exit()
    mlups.append(mlups)

    # Plot results
    np_f = f0.numpy()
    plt.imshow(np_f[3, :, :, 0])
    plt.colorbar()
    plt.savefig("warp_f_.png")
    plt.close()

    ######### Numba #########
    # Make f0, f1
    f0 = cp.ascontiguousarray(cp.empty((n, n, n, 19), dtype=np.float32))
    f1 = cp.ascontiguousarray(cp.empty((n, n, n, 19), dtype=np.float32))

    # Initialize f0
    f0 = numba_initialize_f(f0, dx)

    # Apply streaming and collision
    t0 = time.time()
    for _ in tqdm(range(nr_steps)):
        f0, f1 = numba_apply_collide_stream(f0, f1, tau)
    cp.cuda.Device(0).synchronize()
    t1 = time.time()

    # Compute MLUPS
    mlups = (nr_steps * n * n * n) / (t1 - t0) / 1e6
    backend.append("Numba")
    mlups.append(mlups)

    # Plot results
    np_f = f0
    plt.imshow(np_f[:, :, 0, 3].get())
    plt.colorbar()
    plt.savefig("numba_f_.png")
    plt.close()

    ######### Jax #########
    # Make f0, f1
    f = jnp.zeros((n, n, n, 19), dtype=jnp.float32)

    # Initialize f0
    # f = jax_initialize_f(f, dx)

    # Apply streaming and collision
    t0 = time.time()
    for _ in tqdm(range(nr_steps)):
        f = jax_apply_collide_stream(f, tau)
    t1 = time.time()

    # Compute MLUPS
    mlups = (nr_steps * n * n * n) / (t1 - t0) / 1e6
    backend.append("Jax")
    mlups.append(mlups)

    # Plot results
    np_f = f
    plt.imshow(np_f[:, :, 0, 3])
    plt.colorbar()
    plt.savefig("jax_f_.png")
    plt.close()





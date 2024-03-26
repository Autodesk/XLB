# Simple script to run different boundary conditions with jax and warp backends
import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from typing import Any
import numpy as np
import jax.numpy as jnp
import warp as wp

wp.init()

import xlb

def run_boundary_conditions(backend):

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
    nr = 256
    shape = (nr, nr, nr)
    if backend == "jax":
        grid = xlb.grid.JaxGrid(shape=shape)
    elif backend == "warp":
        grid = xlb.grid.WarpGrid(shape=shape)

    # Make feilds
    f_pre = grid.create_field(cardinality=velocity_set.q, precision=xlb.Precision.FP32)
    f_post = grid.create_field(cardinality=velocity_set.q, precision=xlb.Precision.FP32)
    f = grid.create_field(cardinality=velocity_set.q, precision=xlb.Precision.FP32)
    boundary_id = grid.create_field(cardinality=1, precision=xlb.Precision.UINT8)
    missing_mask = grid.create_field(cardinality=velocity_set.q, precision=xlb.Precision.BOOL)

    # Make needed operators
    equilibrium = xlb.operator.equilibrium.QuadraticEquilibrium(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    equilibrium_bc = xlb.operator.boundary_condition.EquilibriumBC(
        rho=1.0,
        u=(0.0, 0.0, 0.0),
        equilibrium_operator=equilibrium,
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    do_nothing_bc = xlb.operator.boundary_condition.DoNothingBC(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    halfway_bounce_back_bc = xlb.operator.boundary_condition.HalfwayBounceBackBC(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    fullway_bounce_back_bc = xlb.operator.boundary_condition.FullwayBounceBackBC(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    indices_boundary_masker = xlb.operator.boundary_masker.IndicesBoundaryMasker(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )

    # Make indices for boundary conditions (sphere)
    sphere_radius = 32
    x = np.arange(nr)
    y = np.arange(nr)
    z = np.arange(nr)
    X, Y, Z = np.meshgrid(x, y, z)
    indices = np.where(
        (X - nr // 2) ** 2 + (Y - nr // 2) ** 2 + (Z - nr // 2) ** 2
        < sphere_radius**2
    )
    indices = np.array(indices).T
    if backend == "jax":
        indices = jnp.array(indices)
    elif backend == "warp":
        indices = wp.from_numpy(indices, dtype=wp.int32)

    # Test equilibrium boundary condition
    boundary_id, missing_mask = indices_boundary_masker(
        indices,
        equilibrium_bc.id,
        boundary_id,
        missing_mask,
        (0, 0, 0)
    )
    if backend == "jax":
        f = equilibrium_bc(f_pre, f_post, boundary_id, missing_mask)
    elif backend == "warp":
        f = equilibrium_bc(f_pre, f_post, boundary_id, missing_mask, f)
    print(f"Equilibrium BC test passed for {backend}")

    # Test do nothing boundary condition
    boundary_id, missing_mask = indices_boundary_masker(
        indices,
        do_nothing_bc.id,
        boundary_id,
        missing_mask,
        (0, 0, 0)
    )
    if backend == "jax":
        f = do_nothing_bc(f_pre, f_post, boundary_id, missing_mask)
    elif backend == "warp":
        f = do_nothing_bc(f_pre, f_post, boundary_id, missing_mask, f)

    # Test halfway bounce back boundary condition
    boundary_id, missing_mask = indices_boundary_masker(
        indices,
        halfway_bounce_back_bc.id,
        boundary_id,
        missing_mask,
        (0, 0, 0)
    )
    if backend == "jax":
        f = halfway_bounce_back_bc(f_pre, f_post, boundary_id, missing_mask)
    elif backend == "warp":
        f = halfway_bounce_back_bc(f_pre, f_post, boundary_id, missing_mask, f)
    print(f"Halfway bounce back BC test passed for {backend}")

    # Test the full boundary condition
    boundary_id, missing_mask = indices_boundary_masker(
        indices,
        fullway_bounce_back_bc.id,
        boundary_id,
        missing_mask,
        (0, 0, 0)
    )
    if backend == "jax":
        f = fullway_bounce_back_bc(f_pre, f_post, boundary_id, missing_mask)
    elif backend == "warp":
        f = fullway_bounce_back_bc(f_pre, f_post, boundary_id, missing_mask, f)
    print(f"Fullway bounce back BC test passed for {backend}")


if __name__ == "__main__":

    # Test the boundary conditions
    backends = ["warp", "jax"]
    for backend in backends:
        run_boundary_conditions(backend)

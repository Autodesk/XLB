# Simple flow past sphere example using the functional interface to xlb

import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from typing import Any
import numpy as np

import warp as wp

wp.init()

import xlb
from xlb.operator import Operator

class UniformInitializer(Operator):

    def _construct_warp(self):
        # Construct the warp kernel
        @wp.kernel
        def kernel(
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()

            # Set the velocity
            u[0, i, j, k] = 0.0
            u[1, i, j, k] = 0.0
            u[2, i, j, k] = 0.0

            # Set the density
            rho[0, i, j, k] = 1.0

        return None, kernel

    @Operator.register_backend(xlb.ComputeBackend.WARP)
    def warp_implementation(self, rho, u):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                rho,
                u,
            ],
            dim=rho.shape[1:],
        )
        return rho, u


def run_ldc(backend, compute_mlup=True):

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
    rho = grid.create_field(cardinality=1, precision=xlb.Precision.FP32)
    u = grid.create_field(cardinality=velocity_set.d, precision=xlb.Precision.FP32)
    f0 = grid.create_field(cardinality=velocity_set.q, precision=xlb.Precision.FP32)
    f1 = grid.create_field(cardinality=velocity_set.q, precision=xlb.Precision.FP32)
    boundary_id = grid.create_field(cardinality=1, precision=xlb.Precision.UINT8)
    missing_mask = grid.create_field(cardinality=velocity_set.q, precision=xlb.Precision.BOOL)

    # Make operators
    initializer = UniformInitializer(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    collision = xlb.operator.collision.BGK(
        omega=1.9,
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    equilibrium = xlb.operator.equilibrium.QuadraticEquilibrium(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    macroscopic = xlb.operator.macroscopic.Macroscopic(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    stream = xlb.operator.stream.Stream(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    equilibrium_bc = xlb.operator.boundary_condition.EquilibriumBC(
        rho=1.0,
        u=(0, 0.10, 0.0),
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
    half_way_bc = xlb.operator.boundary_condition.HalfwayBounceBackBC(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    full_way_bc = xlb.operator.boundary_condition.FullwayBounceBackBC(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    stepper = xlb.operator.stepper.IncompressibleNavierStokesStepper(
        collision=collision,
        equilibrium=equilibrium,
        macroscopic=macroscopic,
        stream=stream,
        boundary_conditions=[equilibrium_bc, do_nothing_bc, half_way_bc, full_way_bc],
    )
    indices_boundary_masker = xlb.operator.boundary_masker.IndicesBoundaryMasker(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    planar_boundary_masker = xlb.operator.boundary_masker.PlanarBoundaryMasker(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )

    # Set inlet bc (bottom x face)
    lower_bound = (0, 1, 1)
    upper_bound = (0, nr-1, nr-1)
    direction = (1, 0, 0)
    boundary_id, missing_mask = planar_boundary_masker(
        lower_bound,
        upper_bound,
        direction,
        equilibrium_bc.id,
        boundary_id,
        missing_mask,
        (0, 0, 0)
    )

    # Set outlet bc (top x face)
    lower_bound = (nr-1, 1, 1)
    upper_bound = (nr-1, nr-1, nr-1)
    direction = (-1, 0, 0)
    boundary_id, missing_mask = planar_boundary_masker(
        lower_bound,
        upper_bound,
        direction,
        #do_nothing_bc.id,
        full_way_bc.id,
        boundary_id,
        missing_mask,
        (0, 0, 0)
    )

    # Set half way bc (bottom y face)
    lower_bound = (1, 0, 1)
    upper_bound = (nr, 0, nr)
    direction = (0, 1, 0)
    boundary_id, missing_mask = planar_boundary_masker(
        lower_bound,
        upper_bound,
        direction,
        #half_way_bc.id,
        full_way_bc.id,
        boundary_id,
        missing_mask,
        (0, 0, 0)
    )

    # Set half way bc (top y face)
    lower_bound = (1, nr-1, 1)
    upper_bound = (nr, nr-1, nr)
    direction = (0, -1, 0)
    boundary_id, missing_mask = planar_boundary_masker(
        lower_bound,
        upper_bound,
        direction,
        #half_way_bc.id,
        full_way_bc.id,
        boundary_id,
        missing_mask,
        (0, 0, 0)
    )

    # Set half way bc (bottom z face)
    lower_bound = (1, 1, 0)
    upper_bound = (nr, nr, 0)
    direction = (0, 0, 1)
    boundary_id, missing_mask = planar_boundary_masker(
        lower_bound,
        upper_bound,
        direction,
        #half_way_bc.id,
        full_way_bc.id,
        boundary_id,
        missing_mask,
        (0, 0, 0)
    )

    # Set half way bc (top z face)
    lower_bound = (1, 1, nr-1)
    upper_bound = (nr, nr, nr-1)
    direction = (0, 0, -1)
    boundary_id, missing_mask = planar_boundary_masker(
        lower_bound,
        upper_bound,
        direction,
        #half_way_bc.id,
        full_way_bc.id,
        boundary_id,
        missing_mask,
        (0, 0, 0)
    )

    # Set full way bc (sphere)
    """
    sphere_radius = nr // 8
    x = np.arange(nr)
    y = np.arange(nr)
    z = np.arange(nr)
    X, Y, Z = np.meshgrid(x, y, z)
    indices = np.where(
        (X - nr // 2) ** 2 + (Y - nr // 2) ** 2 + (Z - nr // 2) ** 2
        < sphere_radius**2
    )
    indices = np.array(indices).T
    indices = wp.from_numpy(indices, dtype=wp.int32)
    boundary_id, missing_mask = indices_boundary_masker(
        indices,
        full_way_bc.id,
        boundary_id,
        missing_mask,
        (0, 0, 0)
    )
    """

    # Set initial conditions
    if backend == "warp":
        rho, u = initializer(rho, u)
        f0 = equilibrium(rho, u, f0)
    elif backend == "jax":
        rho = rho + 1.0
        f0 = equilibrium(rho, u)

    # Time stepping
    plot_freq = 512
    save_dir = "ldc"
    os.makedirs(save_dir, exist_ok=True)
    num_steps = nr * 32
    start = time.time()

    for _ in tqdm(range(num_steps)):
        # Time step
        if backend == "warp":
            f1 = stepper(f0, f1, boundary_id, missing_mask, _)
            f1, f0 = f0, f1
        elif backend == "jax":
            f0 = stepper(f0, boundary_id, missing_mask, _)

        # Plot if necessary
        if (_ % plot_freq == 0) and (not compute_mlup):
            if backend == "warp":
                rho, u = macroscopic(f0, rho, u)
                local_rho = rho.numpy()
                local_u = u.numpy()
            elif backend == "jax":
                local_rho, local_u = macroscopic(f0)

            # Plot the velocity field, rho and boundary id side by side
            plt.subplot(1, 3, 1)
            plt.imshow(np.linalg.norm(u[:, :, nr // 2, :], axis=0))
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.imshow(rho[0, :, nr // 2, :])
            plt.colorbar()
            plt.subplot(1, 3, 3)
            plt.imshow(boundary_id[0, :, nr // 2, :])
            plt.colorbar()
            plt.savefig(f"{save_dir}/{str(_).zfill(6)}.png")
            plt.close()

    wp.synchronize()
    end = time.time()

    # Print MLUPS
    print(f"MLUPS: {num_steps*nr**3/(end-start)/1e6}")

if __name__ == "__main__":

    # Run the LDC example
    backends = ["warp", "jax"]
    #backends = ["jax"]
    for backend in backends:
        run_ldc(backend, compute_mlup=True)

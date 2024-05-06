# Simple flow past sphere example using the functional interface to xlb

import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from typing import Any
import numpy as np

from xlb.compute_backend import ComputeBackend

import warp as wp

import xlb

xlb.init(
    default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
    default_backend=ComputeBackend.WARP,
    velocity_set=xlb.velocity_set.D2Q9,
)


from xlb.operator import Operator

class UniformInitializer(Operator):

    def _construct_warp(self):
        # Construct the warp kernel
        @wp.kernel
        def kernel(
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            vel: float,
        ):
            # Get the global index
            i, j, k = wp.tid()

            # Set the velocity
            u[0, i, j, k] = vel
            u[1, i, j, k] = 0.0
            u[2, i, j, k] = 0.0

            # Set the density
            rho[0, i, j, k] = 1.0

        return None, kernel

    @Operator.register_backend(xlb.ComputeBackend.WARP)
    def warp_implementation(self, rho, u, vel):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                rho,
                u,
                vel,
            ],
            dim=rho.shape[1:],
        )
        return rho, u


if __name__ == "__main__":
    # Set parameters
    compute_backend = xlb.ComputeBackend.WARP
    precision_policy = xlb.PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D3Q19()

    # Make feilds
    nr = 256
    vel = 0.05
    shape = (nr, nr, nr)
    grid = xlb.grid.grid_factory(shape=shape)
    rho = grid.create_field(cardinality=1)
    u = grid.create_field(cardinality=velocity_set.d, dtype=wp.float32)
    f0 = grid.create_field(cardinality=velocity_set.q, dtype=wp.float32)
    f1 = grid.create_field(cardinality=velocity_set.q, dtype=wp.float32)
    boundary_id_field = grid.create_field(cardinality=1, dtype=wp.uint8)
    missing_mask = grid.create_field(cardinality=velocity_set.q, dtype=wp.bool)

    # Make operators
    initializer = UniformInitializer(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    collision = xlb.operator.collision.BGK(
        omega=1.95,
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
        u=(vel, 0.0, 0.0),
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
    stepper = xlb.operator.stepper.IncompressibleNavierStokesStepper(
        collision=collision,
        equilibrium=equilibrium,
        macroscopic=macroscopic,
        stream=stream,
        equilibrium_bc=equilibrium_bc,
        do_nothing_bc=do_nothing_bc,
        half_way_bc=half_way_bc,
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
    indices = wp.from_numpy(indices, dtype=wp.int32)

    # Set boundary conditions on the indices
    boundary_id_field, missing_mask = indices_boundary_masker(
        indices,
        half_way_bc.id,
        boundary_id_field,
        missing_mask,
        (0, 0, 0)
    )

    # Set inlet bc
    lower_bound = (0, 0, 0)
    upper_bound = (0, nr, nr)
    direction = (1, 0, 0)
    boundary_id_field, missing_mask = planar_boundary_masker(
        lower_bound,
        upper_bound,
        direction,
        equilibrium_bc.id,
        boundary_id_field,
        missing_mask,
        (0, 0, 0)
    )

    # Set outlet bc
    lower_bound = (nr-1, 0, 0)
    upper_bound = (nr-1, nr, nr)
    direction = (-1, 0, 0)
    boundary_id_field, missing_mask = planar_boundary_masker(
        lower_bound,
        upper_bound,
        direction,
        do_nothing_bc.id,
        boundary_id_field,
        missing_mask,
        (0, 0, 0)
    )

    # Set initial conditions
    rho, u = initializer(rho, u, vel)
    f0 = equilibrium(rho, u, f0)

    # Time stepping
    plot_freq = 512
    save_dir = "flow_past_sphere"
    os.makedirs(save_dir, exist_ok=True)
    #compute_mlup = False # Plotting results
    compute_mlup = True
    num_steps = 1024 * 8
    start = time.time()
    for _ in tqdm(range(num_steps)):
        f1 = stepper(f0, f1, boundary_id_field, missing_mask, _)
        f1, f0 = f0, f1
        if (_ % plot_freq == 0) and (not compute_mlup):
            rho, u = macroscopic(f0, rho, u)

            # Plot the velocity field and boundary id side by side
            plt.subplot(1, 2, 1)
            plt.imshow(u[0, :, nr // 2, :].numpy())
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(boundary_id_field[0, :, nr // 2, :].numpy())
            plt.colorbar()
            plt.savefig(f"{save_dir}/{str(_).zfill(6)}.png")
            plt.close()

    wp.synchronize()
    end = time.time()

    # Print MLUPS
    print(f"MLUPS: {num_steps*nr**3/(end-start)/1e6}")

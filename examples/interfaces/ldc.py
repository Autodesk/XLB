# Simple Taylor green example using the functional interface to xlb

import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import warp as wp
wp.init()

import xlb
from xlb.operator import Operator

if __name__ == "__main__":

    # Set parameters
    compute_backend = xlb.ComputeBackend.WARP
    precision_policy = xlb.PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D3Q19()

    # Make feilds
    nr = 256
    shape = (nr, nr, nr)
    grid = xlb.grid.WarpGrid(shape=shape)
    rho = grid.create_field(cardinality=1, dtype=wp.float32)
    u = grid.create_field(cardinality=velocity_set.d, dtype=wp.float32)
    f0 = grid.create_field(cardinality=velocity_set.q, dtype=wp.float32)
    f1 = grid.create_field(cardinality=velocity_set.q, dtype=wp.float32)
    boundary_id = grid.create_field(cardinality=1, dtype=wp.uint8)
    mask = grid.create_field(cardinality=velocity_set.q, dtype=wp.bool)

    # Make operators
    collision = xlb.operator.collision.BGK(
            omega=1.9,
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
     
    stepper = xlb.operator.stepper.IncompressibleNavierStokesStepper(
            collision=collision,
            equilibrium=equilibrium,
            macroscopic=macroscopic,
            stream=stream,
            boundary_conditions=[])

    # Parrallelize the stepper
    #stepper = grid.parallelize_operator(stepper)

    # Set initial conditions
    rho, u = initializer(f0, rho, u, 0.1, nr)
    f0 = equilibrium(rho, u, f0)

    # Time stepping
    plot_freq = 32
    save_dir = "taylor_green"
    os.makedirs(save_dir, exist_ok=True)
    #compute_mlup = False # Plotting results 
    compute_mlup = True
    num_steps = 1024
    start = time.time()
    for _ in tqdm(range(num_steps)):
        f1 = stepper(f0, f1, boundary_id, mask, _)
        f1, f0 = f0, f1
        if (_ % plot_freq == 0) and (not compute_mlup):
            rho, u = macroscopic(f0, rho, u)
            plt.imshow(u[0, :, nr//2, :].numpy())
            plt.colorbar()
            plt.savefig(f"{save_dir}/{str(_).zfill(4)}.png")
            plt.close()
    wp.synchronize()
    end = time.time()

    # Print MLUPS
    print(f"MLUPS: {num_steps*nr**3/(end-start)/1e6}")

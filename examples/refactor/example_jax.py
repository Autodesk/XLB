# from IPython import display
import numpy as np
import jax
import jax.numpy as jnp
import scipy
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import xlb

if __name__ == "__main__":
    # Simulation parameters
    nr = 128
    vel = 0.05
    visc = 0.00001
    omega = 1.0 / (3.0 * visc + 0.5)
    length = 2 * np.pi

    # Geometry (sphere)
    lin = np.linspace(0, length, nr)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    XYZ = np.stack([X, Y, Z], axis=-1)
    radius = np.pi / 8.0

    # XLB precision policy
    precision_policy = xlb.precision_policy.Fp32Fp32()

    # XLB lattice
    velocity_set = xlb.velocity_set.D3Q27()

    # XLB equilibrium
    equilibrium = xlb.operator.equilibrium.QuadraticEquilibrium(velocity_set=velocity_set)

    # XLB macroscopic
    macroscopic = xlb.operator.macroscopic.Macroscopic(velocity_set=velocity_set)

    # XLB collision
    collision = xlb.operator.collision.KBC(omega=omega, velocity_set=velocity_set)

    # XLB stream
    stream = xlb.operator.stream.Stream(velocity_set=velocity_set)

    # XLB noslip boundary condition (sphere)
    in_cylinder = ((X - np.pi/2.0)**2 + (Y - np.pi)**2 + (Z - np.pi)**2) < radius**2
    indices = np.argwhere(in_cylinder)
    bounce_back = xlb.operator.boundary_condition.FullBounceBack.from_indices(
        indices=indices,
        velocity_set=velocity_set
    )

    # XLB outflow boundary condition
    outflow = xlb.operator.boundary_condition.DoNothing.from_indices(
        indices=np.argwhere(XYZ[..., 0] == length),
        velocity_set=velocity_set
    )

    # XLB inflow boundary condition
    inflow = xlb.operator.boundary_condition.EquilibriumBoundary.from_indices(
        indices=np.argwhere(XYZ[..., 0] == 0.0),
        velocity_set=velocity_set,
        rho=1.0,
        u=np.array([vel, 0.0, 0.0]),
        equilibrium=equilibrium
    )

    # XLB stepper
    stepper = xlb.operator.stepper.NSE(
        collision=collision,
        stream=stream,
        equilibrium=equilibrium,
        macroscopic=macroscopic,
        boundary_conditions=[bounce_back, outflow, inflow],
        precision_policy=precision_policy,
    )

    # Make initial condition
    u = jnp.stack([vel * jnp.ones_like(X), jnp.zeros_like(X), jnp.zeros_like(X)], axis=-1)
    rho = jnp.expand_dims(jnp.ones_like(X), axis=-1)
    f = equilibrium(rho, u)

    # Get boundary id and mask
    ijk = jnp.meshgrid(jnp.arange(nr), jnp.arange(nr), jnp.arange(nr), indexing="ij")
    boundary_id, mask = stepper.set_boundary(jnp.stack(ijk, axis=-1))

    # Run simulation
    tic = time.time()
    nr_iter = 4096
    for i in tqdm(range(nr_iter)):
        f = stepper(f, boundary_id, mask, i)

        if i % 32 == 0:
            # Get u, rho from f
            rho, u = macroscopic(f)
            norm_u = jnp.linalg.norm(u, axis=-1)
            norm_u = (1.0 - jnp.minimum(boundary_id, 1.0)) * norm_u

            # Plot
            plt.imshow(norm_u[..., nr//2], cmap="jet")
            plt.colorbar()
            plt.savefig(f"img_{str(i).zfill(5)}.png")
            plt.close()

    # Sync to host
    f = f.block_until_ready()
    toc = time.time()
    print(f"MLUPS: {(nr_iter * nr**3) / (toc - tic) / 1e6}")

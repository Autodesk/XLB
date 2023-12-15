# from IPython import display
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'

import numpy as np
import jax
import jax.numpy as jnp
import scipy
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpi4py import MPI
import cupy as cp

import xlb
from xlb.experimental.ooc import OOCmap, OOCArray

import phantomgaze as pg

comm = MPI.COMM_WORLD

@jax.jit
def q_criterion(u):
    # Compute derivatives
    u_x = u[..., 0]
    u_y = u[..., 1]
    u_z = u[..., 2]

    # Compute derivatives
    u_x_dx = (u_x[2:, 1:-1, 1:-1] - u_x[:-2, 1:-1, 1:-1]) / 2
    u_x_dy = (u_x[1:-1, 2:, 1:-1] - u_x[1:-1, :-2, 1:-1]) / 2
    u_x_dz = (u_x[1:-1, 1:-1, 2:] - u_x[1:-1, 1:-1, :-2]) / 2
    u_y_dx = (u_y[2:, 1:-1, 1:-1] - u_y[:-2, 1:-1, 1:-1]) / 2
    u_y_dy = (u_y[1:-1, 2:, 1:-1] - u_y[1:-1, :-2, 1:-1]) / 2
    u_y_dz = (u_y[1:-1, 1:-1, 2:] - u_y[1:-1, 1:-1, :-2]) / 2
    u_z_dx = (u_z[2:, 1:-1, 1:-1] - u_z[:-2, 1:-1, 1:-1]) / 2
    u_z_dy = (u_z[1:-1, 2:, 1:-1] - u_z[1:-1, :-2, 1:-1]) / 2
    u_z_dz = (u_z[1:-1, 1:-1, 2:] - u_z[1:-1, 1:-1, :-2]) / 2

    # Compute vorticity
    mu_x = u_z_dy - u_y_dz
    mu_y = u_x_dz - u_z_dx
    mu_z = u_y_dx - u_x_dy
    norm_mu = jnp.sqrt(mu_x ** 2 + mu_y ** 2 + mu_z ** 2)

    # Compute strain rate
    s_0_0 = u_x_dx
    s_0_1 = 0.5 * (u_x_dy + u_y_dx)
    s_0_2 = 0.5 * (u_x_dz + u_z_dx)
    s_1_0 = s_0_1
    s_1_1 = u_y_dy
    s_1_2 = 0.5 * (u_y_dz + u_z_dy)
    s_2_0 = s_0_2
    s_2_1 = s_1_2
    s_2_2 = u_z_dz
    s_dot_s = (
        s_0_0 ** 2 + s_0_1 ** 2 + s_0_2 ** 2 +
        s_1_0 ** 2 + s_1_1 ** 2 + s_1_2 ** 2 +
        s_2_0 ** 2 + s_2_1 ** 2 + s_2_2 ** 2
    )

    # Compute omega
    omega_0_0 = 0.0
    omega_0_1 = 0.5 * (u_x_dy - u_y_dx)
    omega_0_2 = 0.5 * (u_x_dz - u_z_dx)
    omega_1_0 = -omega_0_1
    omega_1_1 = 0.0
    omega_1_2 = 0.5 * (u_y_dz - u_z_dy)
    omega_2_0 = -omega_0_2
    omega_2_1 = -omega_1_2
    omega_2_2 = 0.0
    omega_dot_omega = (
        omega_0_0 ** 2 + omega_0_1 ** 2 + omega_0_2 ** 2 +
        omega_1_0 ** 2 + omega_1_1 ** 2 + omega_1_2 ** 2 +
        omega_2_0 ** 2 + omega_2_1 ** 2 + omega_2_2 ** 2
    )

    # Compute q-criterion
    q = 0.5 * (omega_dot_omega - s_dot_s)

    return norm_mu, q


if __name__ == "__main__":
    # Simulation parameters
    nr = 256
    nx = 3 * nr
    ny = nr
    nz = nr
    vel = 0.05
    visc = 0.00001
    omega = 1.0 / (3.0 * visc + 0.5)
    length = 2 * np.pi
    dx = length / (ny - 1)
    radius = np.pi / 3.0

    # OOC parameters
    sub_steps = 8
    sub_nr = 128
    padding = (sub_steps, sub_steps, sub_steps, 0)

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
    # Create a mask function
    def set_boundary_sphere(ijk, boundary_id, mask, id_number):
        # Get XYZ
        XYZ = ijk * dx
        sphere_mask = jnp.linalg.norm(XYZ - length / 2.0, axis=-1) < radius
        boundary_id = boundary_id.at[sphere_mask].set(id_number)
        mask = mask.at[sphere_mask].set(True)
        return boundary_id, mask
    bounce_back = xlb.operator.boundary_condition.FullBounceBack(
        set_boundary=set_boundary_sphere,
        velocity_set=velocity_set
    )

    # XLB outflow boundary condition
    def set_boundary_outflow(ijk, boundary_id, mask, id_number):
        # Get XYZ
        XYZ = ijk * dx
        outflow_mask = XYZ[..., 0] >= (length * 3.0) - dx
        boundary_id = boundary_id.at[outflow_mask].set(id_number)
        mask = mask.at[outflow_mask].set(True)
        return boundary_id, mask
    outflow = xlb.operator.boundary_condition.DoNothing(
        set_boundary=set_boundary_outflow,
        velocity_set=velocity_set
    )

    # XLB inflow boundary condition
    def set_boundary_inflow(ijk, boundary_id, mask, id_number):
        # Get XYZ
        XYZ = ijk * dx
        inflow_mask = XYZ[..., 0] == 0.0
        boundary_id = boundary_id.at[inflow_mask].set(id_number)
        mask = mask.at[inflow_mask].set(True)
        return boundary_id, mask
    inflow = xlb.operator.boundary_condition.EquilibriumBoundary(
        set_boundary=set_boundary_inflow,
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

    # Make OOC arrays
    f = OOCArray(
        shape=(nx, ny, nz, velocity_set.q),
        dtype=np.float32,
        tile_shape=(sub_nr, sub_nr, sub_nr, velocity_set.q),
        padding=padding,
        comm=comm,
        devices=[cp.cuda.Device(0) for i in range(comm.size)],
        codec=None,
        nr_compute_tiles=1,
    )

    camera_radius = length * 2.0
    focal_point = (3.0 * length / 2.0, length / 2.0, length / 2.0)
    angle = 1 * 0.0001
    camera_position = (focal_point[0] + camera_radius * np.sin(angle), focal_point[1], focal_point[2] + camera_radius * np.cos(angle))
    camera = pg.Camera(
        position=camera_position,
        focal_point=focal_point,
        view_up=(0.0, 1.0, 0.0),
        height=1440,
        width=2560,
        max_depth=6.0 * length,
    )
    screen_buffer = pg.ScreenBuffer.from_camera(camera)


    # Initialize f
    @OOCmap(comm, (0,), backend="jax")
    def initialize_f(f):
        # Get inputs
        shape = f.shape[:-1]
        u = jnp.stack([vel * jnp.ones(shape), jnp.zeros(shape), jnp.zeros(shape)], axis=-1)
        rho = jnp.expand_dims(jnp.ones(shape), axis=-1)
        f = equilibrium(rho, u)
        return f
    f = initialize_f(f)

    # Stepping function
    @OOCmap(comm, (0,), backend="jax", add_index=True)
    def ooc_stepper(f):

        # Get tensors
        f, global_index = f

        # Get ijk
        lin_i = jnp.arange(global_index[0], global_index[0] + f.shape[0])
        lin_j = jnp.arange(global_index[1], global_index[1] + f.shape[1])
        lin_k = jnp.arange(global_index[2], global_index[2] + f.shape[2])
        ijk = jnp.meshgrid(lin_i, lin_j, lin_k, indexing="ij")
        ijk = jnp.stack(ijk, axis=-1)

        # Set boundary_id and mask
        boundary_id, mask = stepper.set_boundary(ijk)

        # Run stepper
        for _ in range(sub_steps):
            f = stepper(f, boundary_id, mask, _)

        # Wait till f is computed using jax
        f = f.block_until_ready()

        return f

    # Make a render function
    @OOCmap(comm, (0,), backend="jax", add_index=True)
    def render(f, screen_buffer, camera):

        # Get tensors
        f, global_index = f

        # Get ijk
        lin_i = jnp.arange(global_index[0], global_index[0] + f.shape[0])
        lin_j = jnp.arange(global_index[1], global_index[1] + f.shape[1])
        lin_k = jnp.arange(global_index[2], global_index[2] + f.shape[2])
        ijk = jnp.meshgrid(lin_i, lin_j, lin_k, indexing="ij")
        ijk = jnp.stack(ijk, axis=-1)

        # Set boundary_id and mask
        boundary_id, mask = stepper.set_boundary(ijk)
        sphere = (boundary_id == 1).astype(jnp.float32)[1:-1, 1:-1, 1:-1]

        # Get rho, u
        rho, u = macroscopic(f)

        # Get q-cr
        norm_mu, q = q_criterion(u)

        # Make volumes
        origin = ((global_index[0] + 1) * dx, (global_index[1] + 1) * dx, (global_index[2] + 1) * dx)
        q_volume = pg.objects.Volume(
            q, spacing=(dx, dx, dx), origin=origin
        )
        norm_mu_volume = pg.objects.Volume(
            norm_mu, spacing=(dx, dx, dx), origin=origin
        )
        sphere_volume = pg.objects.Volume(
            sphere, spacing=(dx, dx, dx), origin=origin
        )

        # Render
        screen_buffer = pg.render.contour(
            q_volume,
            threshold=0.000005,
            color=norm_mu_volume,
            colormap=pg.Colormap("jet", vmin=0.0, vmax=0.025),
            camera=camera,
            screen_buffer=screen_buffer,
        )
        screen_buffer = pg.render.contour(
            sphere_volume,
            threshold=0.5,
            camera=camera,
            screen_buffer=screen_buffer,
        )

        return f

    # Run simulation
    tic = time.time()
    nr_iter = 128 * nr // sub_steps
    nr_frames = 1024
    for i in tqdm(range(nr_iter)):
        f = ooc_stepper(f)

        if  i % (nr_iter // nr_frames) == 0:
            # Rotate camera
            camera_radius = length * 1.0
            focal_point = (length / 2.0, length / 2.0, length / 2.0)
            angle = (np.pi / nr_iter) * i
            camera_position = (focal_point[0] + camera_radius * np.sin(angle), focal_point[1], focal_point[2] + camera_radius * np.cos(angle))
            camera = pg.Camera(
                position=camera_position,
                focal_point=focal_point,
                view_up=(0.0, 1.0, 0.0),
                height=1080,
                width=1920,
                max_depth=6.0 * length,
            )

            # Render global setup
            screen_buffer = pg.render.wireframe(
                lower_bound=(0.0, 0.0, 0.0),
                upper_bound=(3.0*length, length, length),
                thickness=length/100.0,
                camera=camera,
            )
            screen_buffer = pg.render.axes(
                size=length/30.0,
                center=(0.0, 0.0, length*1.1),
                camera=camera,
                screen_buffer=screen_buffer
            )

            # Render
            render(f, screen_buffer, camera)

            # Save image
            plt.imsave('./q_criterion_' + str(i).zfill(7) + '.png', np.minimum(screen_buffer.image.get(), 1.0))

    # Sync to host
    cp.cuda.runtime.deviceSynchronize()
    toc = time.time()
    print(f"MLUPS: {(sub_steps * nr_iter * nr**3) / (toc - tic) / 1e6}")

# Simple example of functions to generate a warp kernel for LBM

import warp as wp
import numpy as np

# Initialize Warp
wp.init()

def make_warp_kernel(
    velocity_weight,
    velocity_set,
    dtype=wp.float32,
    dim=3, # slightly hard coded for 3d right now
    q=19,
):

    # Make needed vector classes
    lattice_vec = wp.vec(q, dtype=dtype)
    velocity_vec = wp.vec(dim, dtype=dtype)

    # Make array type
    if dim == 2:
        array_type = wp.array3d(dtype=dtype)
    elif dim == 3:
        array_type = wp.array4d(dtype=dtype)

    # Make everything constant
    velocity_weight = wp.constant(velocity_weight)
    velocity_set = wp.constant(velocity_set)
    q = wp.constant(q)
    dim = wp.constant(dim)

    # Make function for computing exu
    @wp.func
    def compute_exu(u: velocity_vec):
        exu = lattice_vec()
        for _ in range(q):
            for d in range(dim):
                if velocity_set[_, d] == 1:
                    exu[_] += u[d]
                elif velocity_set[_, d] == -1:
                    exu[_] -= u[d]
        return exu

    # Make function for computing feq
    @wp.func
    def compute_feq(
        p: dtype,
        uxu: dtype,
        exu: lattice_vec,
    ):
        factor_1 = 1.5
        factor_2 = 4.5
        feq = lattice_vec()
        for _ in range(q):
            feq[_] = (
                velocity_weight[_] * p * (
                    1.0
                    + factor_1 * (2.0 * exu[_] - uxu)
                    + factor_2 * exu[_] * exu[_]
                )
            )
        return feq

    # Make function for computing u and p
    @wp.func
    def compute_u_and_p(f: lattice_vec):
        p = wp.float32(0.0)
        u = velocity_vec()
        for d in range(dim):
            u[d] = wp.float32(0.0)
        for _ in range(q):
            p += f[_]
            for d in range(dim):
                if velocity_set[_, d] == 1:
                    u[d] += f[_]
                elif velocity_set[_, d] == -1:
                    u[d] -= f[_]
        u /= p
        return u, p

    # bc function
    @wp.func
    def bc_0(pre_f: lattice_vec, post_f: lattice_vec):
        return pre_f
    @wp.func
    def bc_1(pre_f: lattice_vec, post_f: lattice_vec):
        return post_f
    tup_bc = tuple([bc_0, bc_1])
    single_bc = bc_0
    for bc in tup_bc:
        def make_bc(bc, prev_bc):
            @wp.func
            def _bc(pre_f: lattice_vec, post_f: lattice_vec):
                pre_f = prev_bc(pre_f, post_f)
                post_f = single_bc(pre_f, post_f)
                return bc(pre_f, post_f)
            return _bc
        single_bc = make_bc(bc, single_bc)

    # Make function for getting stream index
    @wp.func
    def get_streamed_index(
            i: int,
            x: int,
            y: int,
            z: int,
            width: int,
            height: int,
            length: int,
    ):
        streamed_x = x + velocity_set[i, 0]
        streamed_y = y + velocity_set[i, 1]
        streamed_z = z + velocity_set[i, 2]
        if streamed_x == -1: # TODO hacky
            streamed_x = width - 1
        if streamed_y == -1:
            streamed_y = height - 1
        if streamed_z == -1:
            streamed_z = length - 1
        if streamed_x == width:
            streamed_x = 0
        if streamed_y == height:
            streamed_y = 0
        if streamed_z == length:
            streamed_z = 0
        return streamed_x, streamed_y, streamed_z

    # Make kernel for stream and collide
    @wp.kernel
    def collide_stream(
        f0: array_type,
        f1: array_type,
        width: int,
        height: int,
        length: int,
        tau: float,
    ):

        # Get indices (TODO: no good way to do variable dimension indexing)
        f = lattice_vec()
        x, y, z = wp.tid()
        for i in range(q):
            f[i] = f0[i, x, y, z]

        # Compute p and u
        u, p = compute_u_and_p(f)

        # get uxu
        uxu = wp.dot(u, u)

        # Compute velocity_set dot u
        exu = compute_exu(u)

        # Compute equilibrium
        feq = compute_feq(p, uxu, exu)

        # Set bc
        if x == 0:
            #tup_bc[0](feq, f)
            bc_0(feq, f)
        if x == width - 1:
            bc_1(feq, f)
            #tup_bc[1](feq, f)

        # Set value
        new_f = f - (f - feq) / tau
        for i in range(q):
            (streamed_x, streamed_y, streamed_z) = get_streamed_index(
                i, x, y, z, width, height, length
            )
            f1[i, streamed_x, streamed_y, streamed_z] = new_f[i]

    # make kernel for initialization
    @wp.kernel
    def initialize_taylor_green(
        f0: array_type,
        dx: float,
        vel: float,
        width: int,
        height: int,
        length: int,
        tau: float,
    ):

        # Get indices (TODO: no good way to do variable dimension indexing)
        i, j, k = wp.tid()

        # Get real coordinates
        x = wp.float(i) * dx
        y = wp.float(j) * dx
        z = wp.float(k) * dx

        # Compute velocity
        u = velocity_vec()
        u[0] = vel * wp.sin(x) * wp.cos(y) * wp.cos(z)
        u[1] = -vel * wp.cos(x) * wp.sin(y) * wp.cos(z)
        u[2] = 0.0

        # Compute p
        p = (
            3.0
            * vel
            * vel
            * (1.0 / 16.0)
            * (
                wp.cos(2.0 * x)
                + wp.cos(2.0 * y)
                + wp.cos(2.0 * z)
            )
            + 1.0
        )

        # Compute uxu
        uxu = wp.dot(u, u)

        # Compute velocity_set dot u
        exu = compute_exu(u)

        # Compute equilibrium
        feq = compute_feq(p, uxu, exu)

        # Set value
        for _ in range(q):
            f0[_, i, j, k] = feq[_]

    return collide_stream, initialize_taylor_green

def plt_f(f):
    import matplotlib.pyplot as plt
    plt.imshow(f.numpy()[3, :, :, f.shape[3] // 4])
    plt.show()

if __name__ == "__main__":

    # Parameters
    n = 256
    tau = 0.505
    dim = 3
    q = 19
    lattice_dtype = wp.float32
    lattice_vec = wp.vec(q, dtype=lattice_dtype)

    # Make arrays
    f0 = wp.empty((q, n, n, n), dtype=lattice_dtype, device="cuda:0")
    f1 = wp.empty((q, n, n, n), dtype=lattice_dtype, device="cuda:0")

    # Make velocity set
    velocity_weight = wp.vec(q, dtype=lattice_dtype)(
        [1.0/3.0] + [1.0/18.0] * 6 + [1.0/36.0] * 12
    )
    velocity_set = wp.mat((q, dim), dtype=wp.int32)(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            [0, 1, 1],
            [0, -1, -1],
            [0, 1, -1],
            [0, -1, 1],
            [1, 0, 1],
            [-1, 0, -1],
            [1, 0, -1],
            [-1, 0, 1],
            [1, 1, 0],
            [-1, -1, 0],
            [1, -1, 0],
            [-1, 1, 0],
        ]
    )

    # Make kernel
    collide_stream, initialize = make_warp_kernel(
        velocity_weight,
        velocity_set,
        dtype=lattice_dtype,
        dim=dim,
        q=q,
    )

    # Initialize
    cs = 1.0 / np.sqrt(3.0)
    vel = 0.1 * cs
    dx = 2.0 * np.pi / n
    wp.launch(
        initialize,
        inputs=[
            f0,
            dx,
            vel,
            n,
            n,
            n,
            tau,
        ],
        dim=(n, n, n),
    )

    # Compute MLUPS
    import time
    import tqdm
    nr_iterations = 128
    start = time.time()
    for i in tqdm.tqdm(range(nr_iterations)):
        #if i % 10 == 0:
        #    plt_f(f0)

        wp.launch(
            collide_stream,
            inputs=[
                f0,
                f1,
                n,
                n,
                n,
                tau,
            ],
            dim=(n, n, n),
        )
        f0, f1 = f1, f0
    wp.synchronize()
    end = time.time()
    print("MLUPS: ", (nr_iterations * n * n * n) / (end - start) / 1e6)

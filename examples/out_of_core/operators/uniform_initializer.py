from typing import Any
import warp as wp


class UniformInitializer:
    # NOTE: This could be overridden to use more complex initializers

    def __init__(
        self,
        initial_rho,
        initial_u,
    ):
        self.initial_rho = initial_rho
        self.initial_u = initial_u

    @wp.kernel
    def uniform_initializer_kernel(
        rho: wp.array4d(dtype=Any),
        u: wp.array4d(dtype=Any),
        boundary_id: wp.array4d(dtype=wp.uint8),
        initial_u: wp.vec3f,
        initial_rho: float,
    ):
        # Get the global index
        i, j, k = wp.tid()

        # Set the velocity
        u[0, i, j, k] = initial_u[0]
        u[1, i, j, k] = initial_u[1]
        u[2, i, j, k] = initial_u[2]

        # Set the density
        rho[0, i, j, k] = initial_rho

    def __call__(self, rho, u, boundary_id):
        # Launch the warp kernel
        wp.launch(
            self.uniform_initializer_kernel,
            inputs=[rho, u, boundary_id, wp.vec3f(self.initial_u), self.initial_rho],
            dim=rho.shape[1:],
        )
        return rho, u

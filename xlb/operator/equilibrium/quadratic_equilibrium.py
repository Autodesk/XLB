import jax.numpy as jnp
from jax import jit
from xlb.velocity_set import VelocitySet
from xlb.compute_backends import ComputeBackends
from xlb.operator.equilibrium.equilibrium import Equilibrium
from functools import partial
from xlb.operator import Operator
from xlb.global_config import GlobalConfig


class QuadraticEquilibrium(Equilibrium):
    """
    Quadratic equilibrium of Boltzmann equation using hermite polynomials.
    Standard equilibrium model for LBM.

    TODO: move this to a separate file and lower and higher order equilibriums
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        compute_backend=None,
    ):
        velocity_set = velocity_set or GlobalConfig.velocity_set
        compute_backend = compute_backend or GlobalConfig.compute_backend

        super().__init__(velocity_set, compute_backend)

    @Operator.register_backend(ComputeBackends.JAX)
    @partial(jit, static_argnums=(0), donate_argnums=(1, 2))
    def jax_implementation(self, rho, u):
        cu = 3.0 * jnp.tensordot(self.velocity_set.c, u, axes=(0, 0))
        usqr = 1.5 * jnp.sum(jnp.square(u), axis=0, keepdims=True)
        w = self.velocity_set.w.reshape((-1,) + (1,) * (len(rho.shape) - 1))

        feq = rho * w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
        return feq

    @Operator.register_backend(ComputeBackends.PALLAS)
    def pallas_implementation(self, rho, u):
        u0, u1, u2 = u[0], u[1], u[2]
        usqr = 1.5 * (u0**2 + u1**2 + u2**2)

        eq = [
            rho[0] * (1.0 / 18.0) * (1.0 - 3.0 * u0 + 4.5 * u0 * u0 - usqr),
            rho[0] * (1.0 / 18.0) * (1.0 - 3.0 * u1 + 4.5 * u1 * u1 - usqr),
            rho[0] * (1.0 / 18.0) * (1.0 - 3.0 * u2 + 4.5 * u2 * u2 - usqr),
        ]

        combined_velocities = [u0 + u1, u0 - u1, u0 + u2, u0 - u2, u1 + u2, u1 - u2]

        for vel in combined_velocities:
            eq.append(
                rho[0] * (1.0 / 36.0) * (1.0 - 3.0 * vel + 4.5 * vel * vel - usqr)
            )

        eq.append(rho[0] * (1.0 / 3.0) * (1.0 - usqr))

        for i in range(3):
            eq.append(eq[i] + rho[0] * (1.0 / 18.0) * 6.0 * u[i])

        for i, vel in enumerate(combined_velocities, 3):
            eq.append(eq[i] + rho[0] * (1.0 / 36.0) * 6.0 * vel)

        return jnp.array(eq)

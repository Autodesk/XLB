import jax.numpy as jnp
from jax import jit
from xlb.velocity_set import VelocitySet
from xlb.compute_backends import ComputeBackends
from xlb.operator.collision.collision import Collision
from xlb.operator import Operator
from functools import partial


class BGK(Collision):
    """
    BGK collision operator for LBM.
    """

    def __init__(
        self,
        omega: float,
        velocity_set: VelocitySet = None,
        compute_backend=None,
    ):
        super().__init__(
            omega=omega, velocity_set=velocity_set, compute_backend=compute_backend
        )

    @Operator.register_backend(ComputeBackends.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(self, f: jnp.ndarray, feq: jnp.ndarray):
        fneq = f - feq
        fout = f - self.omega * fneq
        return fout

    @Operator.register_backend(ComputeBackends.WARP)
    def warp_implementation(self, *args, **kwargs):
        # Implementation for the Warp backend
        raise NotImplementedError

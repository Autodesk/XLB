import jax.numpy as jnp
from jax import jit, device_count
import jax.lax as lax
from functools import partial
import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.equilibrium.equilibrium import Equilibrium
from xlb.operator.boundary_condition.boundary_condition import (
    BoundaryCondition,
    ImplementationStep,
)
from xlb.operator.boundary_condition.boundary_masker import (
    BoundaryMasker,
    IndicesBoundaryMasker,
)



class EquilibriumBoundary(BoundaryCondition):
    """
    Equilibrium boundary condition for a lattice Boltzmann method simulation.
    """

    def __init__(
        self,
        set_boundary,
        rho: float,
        u: tuple[float, float],
        equilibrium: Equilibrium,
        boundary_masker: BoundaryMasker,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend,
    ):
        super().__init__(
            ImplementationStep.COLLISION,
            implementation_step=ImplementationStep.STREAMING,
            velocity_set=velocity_set,
            compute_backend=compute_backend,
        )
        self.f = equilibrium(rho, u)

    @classmethod
    def from_indices(
        cls,
        indices: np.ndarray,
        rho: float,
        u: tuple[float, float],
        equilibrium: Equilibrium,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend,
    ):
        """
        Creates a boundary condition from a list of indices.
        """

        return cls(
            set_boundary=cls._set_boundary_from_indices(indices),
            rho=rho,
            u=u,
            equilibrium=equilibrium,
            velocity_set=velocity_set,
            compute_backend=compute_backend,
        )

    @partial(jit, static_argnums=(0), donate_argnums=(1, 2, 3, 4))
    def apply_jax(self, f_pre, f_post, boundary, mask):
        equilibrium_mask = jnp.repeat(boundary[..., None], self.velocity_set.q, axis=-1)
        equilibrium_f = jnp.repeat(self.f[None, ...], boundary.shape[0], axis=0)
        equilibrium_f = jnp.repeat(equilibrium_f[:, None], boundary.shape[1], axis=1)
        equilibrium_f = jnp.repeat(equilibrium_f[:, :, None], boundary.shape[2], axis=2)
        f = lax.select(equilibrium_mask, equilibrium_f, f_post)
        return f

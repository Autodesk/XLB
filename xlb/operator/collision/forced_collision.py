import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.collision.collision import Collision
from xlb.operator import Operator
from functools import partial
from xlb.operator.force import ExactDifference


class ForcedCollision(Collision):
    """
    A collision operator for LBM with external force.
    """

    def __init__(
        self,
        collision_operator: Operator,
        forcing_scheme="exact_difference",
        force_vector=None,
    ):
        assert collision_operator is not None
        self.collision_operator = collision_operator
        super().__init__()

        assert forcing_scheme == "exact_difference", NotImplementedError(f"Force model {forcing_scheme} not implemented!")
        assert force_vector.shape[0] == self.velocity_set.d, "Check the dimensions of the input force!"
        self.force_vector = force_vector
        if forcing_scheme == "exact_difference":
            self.forcing_operator = ExactDifference(force_vector)

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(self, f: jnp.ndarray, feq: jnp.ndarray, rho, u, omega):
        fout = self.collision_operator(f, feq, rho, u, omega)
        fout = self.forcing_operator(fout, feq, rho, u)
        return fout

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _u_vec = wp.vec(self.velocity_set.d, dtype=self.compute_dtype)
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)

        # Construct the functional
        @wp.func
        def functional(f: Any, feq: Any, rho: Any, u: Any, omega: Any):
            fout = self.collision_operator.warp_functional(f, feq, rho, u, omega)
            fout = self.forcing_operator.warp_functional(fout, feq, rho, u)
            return fout

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            feq: wp.array4d(dtype=Any),
            fout: wp.array4d(dtype=Any),
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            omega: Any,
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)  # TODO: Warp needs to fix this

            # Load needed values
            _f = _f_vec()
            _feq = _f_vec()
            _d = self.velocity_set.d
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
                _feq[l] = feq[l, index[0], index[1], index[2]]
            _u = _u_vec()
            for l in range(_d):
                _u[l] = u[l, index[0], index[1], index[2]]
            _rho = rho[0, index[0], index[1], index[2]]

            # Compute the collision
            _fout = functional(_f, _feq, _rho, _u, omega)

            # Write the result
            for l in range(self.velocity_set.q):
                fout[l, index[0], index[1], index[2]] = _fout[l]

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, feq, fout, rho, u, omega):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f,
                feq,
                fout,
                rho,
                u,
                omega,
            ],
            dim=f.shape[1:],
        )
        return fout

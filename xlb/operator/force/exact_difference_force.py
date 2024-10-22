from functools import partial
from jax import jit, lax
import warp as wp
from typing import Any

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.equilibrium import QuadraticEquilibrium


class ExactDifference(Operator):
    """
    Add external body force based on the exact-difference method due to Kupershtokh (2004)

    References
    ----------
    Kupershtokh, A. (2004). New method of incorporating a body force term into the lattice Boltzmann equation. In
    Proceedings of the 5th International EHD Workshop (pp. 241-246). University of Poitiers, Poitiers, France.
    Chikatamarla, S. S., & Karlin, I. V. (2013). Entropic lattice Boltzmann method for turbulent flow simulations:
    Boundary conditions. Physica A, 392, 1925-1930.
    Krüger, T., et al. (2017). The lattice Boltzmann method. Springer International Publishing, 10.978-3, 4-15.
    """

    def __init__(
        self,
        force_vector,
        equilibrium: Operator = None,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        # TODO: currently we are limited to a single force vector not a spatially dependent forcing field
        self.force_vector = force_vector
        self.equilibrium = QuadraticEquilibrium() if equilibrium is None else equilibrium

        # Call the parent constructor
        super().__init__(
            velocity_set,
            precision_policy,
            compute_backend,
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_postcollision, feq, rho, u):
        """
        Parameters
        ----------
        f_postcollision: jax.numpy.ndarray
            The post-collision distribution functions.
        feq: jax.numpy.ndarray
            The equilibrium distribution functions.
        rho: jax.numpy.ndarray
            The density field.

        u: jax.numpy.ndarray
            The velocity field.

        Returns
        -------
        f_postcollision: jax.numpy.ndarray
        The post-collision distribution functions with the force applied.
        """
        delta_u = lax.broadcast_in_dim(self.force_vector, u.shape, (0,))
        feq_force = self.equilibrium(rho, u + delta_u)
        f_postcollision += feq_force - feq
        return f_postcollision

    def _construct_warp(self):
        _d = self.velocity_set.d
        _u_vec = wp.vec(_d, dtype=self.compute_dtype)
        if _d == 2:
            _force = _u_vec(self.force_vector[0], self.force_vector[1])
        else:
            _force = _u_vec(self.force_vector[0], self.force_vector[1], self.force_vector[2])

        # Construct the functional
        @wp.func
        def functional(f_postcollision: Any, feq: Any, rho: Any, u: Any):
            delta_u = _force
            feq_force = self.equilibrium.warp_functional(rho, u + delta_u)
            f_postcollision += feq_force - feq
            return f_postcollision

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f_postcollision: Any,
            feq: Any,
            fout: wp.array4d(dtype=Any),
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)  # TODO: Warp needs to fix this

            # Load needed values
            _u = _u_vec()
            for l in range(_d):
                _u[l] = u[l, index[0], index[1], index[2]]
            _rho = rho[0, index[0], index[1], index[2]]

            # Compute the collision
            _fout = functional(f_postcollision, feq, _rho, _u)

            # Write the result
            for l in range(self.velocity_set.q):
                fout[l, index[0], index[1], index[2]] = self.store_dtype(_fout[l])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_postcollision, feq, fout, rho, u):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_postcollision, feq, fout, rho, u],
            dim=f_postcollision.shape[1:],
        )
        return fout

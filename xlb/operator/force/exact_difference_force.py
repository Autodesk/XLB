from functools import partial
from jax import jit, lax
import warp as wp
from typing import Any

from xlb import DefaultConfig
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.equilibrium import QuadraticEquilibrium


class ExactDifference(Operator):
    """
    Add external body force based on the exact-difference method due to Kupershtokh (2004)

    On JAX, the force can either be fixed at construction time or supplied fresh on every
    call via the ``force_vector`` argument to ``jax_implementation`` -- the latter is
    required for a force that depends on another field (e.g. Boussinesq buoyancy driven by
    a coupled temperature solver), since reassigning ``self.force_vector`` after
    construction has no effect once the JIT-compiled call has been traced once. The Warp
    path only supports a single vector fixed at construction time.

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
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
    ):
        # Now the class accept a force vector at construction time, which is used for the Warp backend and as a default for JAX.
        self.force_vector = force_vector

        # Resolve compute_backend the same way Operator.__init__ does, so we
        # know which equilibrium backend to use before super().__init__ runs.
        # Neon kernels reuse Warp functionals, so sub-operators on the Neon
        # backend are built on Warp.
        resolved_backend = compute_backend or DefaultConfig.default_backend
        eq_backend = ComputeBackend.WARP if resolved_backend == ComputeBackend.NEON else resolved_backend
        self.equilibrium = QuadraticEquilibrium(compute_backend=eq_backend)

        # Call the parent constructor
        super().__init__(
            velocity_set,
            precision_policy,
            compute_backend,
        )

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, f_postcollision, feq, rho, u, force_vector=None):
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

        force_vector: jax.numpy.ndarray, optional
            Body force for this call, overriding the vector fixed at construction time.
            Passing it here (rather than reassigning ``self.force_vector``) is what makes
            a time- or field-dependent force (e.g. Boussinesq buoyancy from a coupled
            temperature field) actually take effect: ``self.force_vector`` is baked into
            the JIT trace as a compile-time constant because ``self`` is a static
            argument, so mutating it after the first call has no effect on subsequent
            calls with the same shapes. A traced argument is retraced/updated correctly
            on every call, at no extra compilation cost as long as its shape and dtype
            stay fixed. Defaults to the vector given at construction time.

        Returns
        -------
        f_postcollision: jax.numpy.ndarray
        The post-collision distribution functions with the force applied.
        """
        if force_vector is None:
            force_vector = self.force_vector
        if force_vector.ndim == 1:
            delta_u = lax.broadcast_in_dim(force_vector, u.shape, (0,))
        else:
            assert force_vector.shape == u.shape, f"force_vector has wrong shape {force_vector.shape}, expected {u.shape} or ({u.shape[0]},)"
            delta_u = force_vector
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

    def _construct_neon(self):
        # The neon backend relies on the warp functionals for its operations.
        functional, _ = self._construct_warp()
        return functional, None

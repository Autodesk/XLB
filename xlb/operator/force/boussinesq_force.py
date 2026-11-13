"""
Boussinesq buoyancy force driven by a transported scalar.

Closes the two-way coupling between the scalar (thermal) solver and the flow solver: the
scalar recovered from the thermal populations is turned into a body force field, which is
then handed to :class:`~xlb.operator.stepper.IncompressibleNavierStokesStepper` through its
``force_vector`` argument.
"""

from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator


class BoussinesqForce(Operator):
    r"""Body force of the Boussinesq approximation.

    For a fluid whose density depends linearly on a scalar :math:`\phi` (the temperature
    or a concentration), the buoyancy force per unit mass is

    .. math::

        \mathbf{F} = \mathbf{b} \, (\phi - \phi_{ref}), \qquad \mathbf{b} = -\beta \, \mathbf{g}

    with :math:`\beta` the expansion coefficient and :math:`\mathbf{g}` the gravitational
    acceleration, so :math:`\mathbf{b}` points in the direction a warm parcel accelerates
    (upwards for a fluid that expands on heating). In lattice units, with
    :math:`\rho \simeq 1`, :math:`\mathbf{F}` is directly the velocity increment per step
    used by :class:`~xlb.operator.force.ExactDifference`. A common choice is
    :math:`|\mathbf{b}| \, \Delta\phi = Ri \, U^2 / L` for a Richardson number :math:`Ri`.

    Parameters
    ----------
    buoyancy : array-like
        Buoyancy vector :math:`\mathbf{b}` of length ``d``, in lattice units.
    reference_value : float, optional
        Reference scalar :math:`\phi_{ref}` at which the force vanishes. Defaults to ``0.0``.

    Notes
    -----
    On JAX the operator returns a new ``(d, *grid)`` array; on Warp it writes into a
    preallocated field of cardinality ``d`` (e.g. ``grid.create_field(cardinality=d)``), so
    the force never leaves the device.
    """

    def __init__(
        self,
        buoyancy,
        reference_value: float = 0.0,
        velocity_set=None,
        precision_policy=None,
        compute_backend=None,
    ):
        self.buoyancy = np.asarray(buoyancy, dtype=np.float64)
        self.reference_value = float(reference_value)
        assert self.buoyancy.ndim == 1, "buoyancy must be a vector of length d"
        super().__init__(velocity_set, precision_policy, compute_backend)
        assert self.buoyancy.shape[0] == self.velocity_set.d, f"buoyancy has length {self.buoyancy.shape[0]}, expected {self.velocity_set.d}"

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(self, phi):
        # phi has cardinality 1, shape (1, *grid); the force has shape (d, *grid)
        buoyancy = jnp.asarray(self.buoyancy, dtype=self.compute_dtype).reshape((-1,) + (1,) * (phi.ndim - 1))
        force = buoyancy * (self.precision_policy.cast_to_compute_jax(phi) - self.reference_value)
        return self.precision_policy.cast_to_store_jax(force)

    def _construct_warp(self):
        _d = self.velocity_set.d
        _b = [self.compute_dtype(b) for b in self.buoyancy]
        _phi_ref = self.compute_dtype(self.reference_value)
        _u_vec = wp.types.vector(length=_d, dtype=self.compute_dtype)
        if _d == 2:
            _buoyancy = _u_vec(_b[0], _b[1])
        else:
            _buoyancy = _u_vec(_b[0], _b[1], _b[2])

        # Construct the functional
        @wp.func
        def functional(phi: Any):
            return _buoyancy * (phi - _phi_ref)

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            phi: wp.array4d(dtype=Any),
            force: wp.array4d(dtype=Any),
        ):
            i, j, k = wp.tid()
            _force = functional(self.compute_dtype(phi[0, i, j, k]))
            for d in range(_d):
                force[d, i, j, k] = self.store_dtype(_force[d])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, phi, force):
        wp.launch(
            self.warp_kernel,
            inputs=[phi, force],
            dim=phi.shape[1:],
            device=phi.device,
        )
        return force

    @Operator.register_backend(ComputeBackend.NEON)
    def neon_implementation(self, phi, force):
        raise NotImplementedError("This feature is not implemented in XLB with the NEON backend yet.")

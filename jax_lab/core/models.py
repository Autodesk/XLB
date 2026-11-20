import jax.numpy as jnp
from jax import jit
import numpy as np
from functools import partial
from .base import Components, LBMBase, _lattice_dot, _lattice_sum, component_mode, lattice_density_momentum, matvec_components, stack_components
from .lattice import LatticeD2Q9, LatticeD3Q19, LatticeD3Q27

"""
Collision operators are defined in this file for different models.
"""


def _direction_tables(lattice):
    """Lattice directions (as integer tuples) and weights (as floats), compile-time constants for per-direction collisions."""
    c = np.asarray(lattice.c)
    return tuple(tuple(int(v) for v in c[:, i]) for i in range(lattice.q)), tuple(float(v) for v in np.asarray(lattice.w))


def _per_direction_moments(lattice, f):
    """Populations as per-direction fields, with the density and velocity components summed from them."""
    directions, weights = _direction_tables(lattice)
    channels = [f[..., i] for i in range(lattice.q)]
    rho = sum(channels)
    velocity = [_lattice_sum(directions, channels, axis) / rho for axis in range(lattice.d)]
    return directions, weights, channels, rho, velocity


def _is_default_method(obj, name, default):
    """Whether obj uses the given default method: not overridden by its class, nor replaced on the instance (as
    Thermal does with fluid_solver.apply_force)."""
    return getattr(type(obj), name) is default and name not in vars(obj)


def _per_direction_force(du, dim, velocity):
    """Per-axis fields of the body-force velocity increment du and the scalar 1.5 (2 u.du + du.du) of the exact-difference
    forcing, as used by LBMBase.apply_force; None when there is no force."""
    if du is None:
        return None, None
    du = [du[..., axis] for axis in range(dim)]
    delta_usqr = 1.5 * (2.0 * sum(v * d for v, d in zip(velocity, du, strict=True)) + sum(d * d for d in du))
    return du, delta_usqr


class BGKSim(LBMBase):
    """
    BGK simulation class.

    This class implements the Bhatnagar-Gross-Krook (BGK) approximation for the collision step in the Lattice Boltzmann Method.
    """

    # Direction-major populations in run() by default in 3D (+32%; see LBMBase._direction_major_step); not measured in 2D
    _direction_major_dims = (3,)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,))
    def collision(self, f):
        """
        BGK collision step for lattice.

        The collision step is where the main physics of the LBM is applied. In the BGK approximation,
        the distribution function is relaxed towards the equilibrium distribution function.
        """
        f = self.precision_policy.cast_to_compute(f)
        if self._uses_default_bgk_methods():
            return self._collision_components(f)
        rho, u = self.update_macroscopic(f)
        feq = self.equilibrium(rho, u, cast_output=False)
        fneq = f - feq
        fout = f - self.omega * fneq
        if self.force is not None:
            fout = self.apply_force(fout, feq, rho, u)
        return self.precision_policy.cast_to_output(fout)

    def _collision_applies_bounceback(self):
        """The per-direction collision applies the full-way bounce-back itself (see LBMBase.apply_fused_bounceback)."""
        return type(self).collision is BGKSim.collision and self._uses_default_bgk_methods()

    def _uses_default_bgk_methods(self):
        """Whether the macroscopic fields, equilibrium and forcing are the defaults (then the per-direction collision applies)."""
        cls = type(self)
        return (
            cls.update_macroscopic is LBMBase.update_macroscopic
            and cls.equilibrium is LBMBase.equilibrium
            and _is_default_method(self, "apply_force", LBMBase.apply_force)
        )

    def _collision_components(self, f):
        """
        The BGK collision with exact-difference forcing evaluated per lattice direction, for the default macroscopic fields,
        equilibrium and forcing: fout_i = f_i - omega (f_i - feq_i) + delta_feq_i.

        Every term is computed per direction with the lattice velocities as compile-time constants, so XLA fuses the moments,
        equilibrium, forcing and relaxation into one kernel with one thread per node (see LBMBase.stack_populations). Same
        formulas as the array path; results can differ at roundoff level (summation order of the moments).
        """
        directions, weights, channels, rho, velocity = _per_direction_moments(self.lattice, f)
        usqr = 1.5 * sum(v * v for v in velocity)
        du, delta_usqr = _per_direction_force(self.get_force_field(rho[..., None]), self.dim, velocity) if self.force is not None else (None, None)
        outputs = []
        for direction, w, fi in zip(directions, weights, channels, strict=True):
            cu = 3.0 * _lattice_dot(direction, velocity)
            rho_w = rho * w
            feq = rho_w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
            out = fi - self.omega * (fi - feq)
            if du is not None:
                dcu = 3.0 * _lattice_dot(direction, du)
                out = out + rho_w * (dcu * (1.0 + cu + 0.5 * dcu) - delta_usqr)
            outputs.append(out)
        outputs = self.apply_fused_bounceback(outputs, channels, self.bounceback_mask)
        return self.stack_populations([self.precision_policy.cast_to_output(out) for out in outputs])


class KBCSim(LBMBase):
    """
    KBC simulation class.

    This class implements the Karlin-Bösch-Chikatamarla (KBC) model for the collision step in the Lattice Boltzmann Method.
    """

    # Direction-major populations in run() by default (with the per-direction collision: +4% in 3D, +7% in 2D;
    # see LBMBase._direction_major_step)
    _direction_major_dims = (2, 3)

    def __init__(self, **kwargs):
        if kwargs.get("lattice").name != "D3Q27" and kwargs.get("nz") > 0:
            raise ValueError("KBC collision operator in 3D must only be used with D3Q27 lattice.")
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,))
    def collision(self, f):
        """
        KBC collision step for lattice.
        """
        f = self.precision_policy.cast_to_compute(f)
        if self._uses_default_kbc_methods():
            return self._collision_components(f)
        tiny = 1e-32
        beta = self.omega * 0.5
        rho, u = self.update_macroscopic(f)
        feq = self.equilibrium(rho, u, cast_output=False)
        fneq = f - feq
        if self.dim == 2:
            deltaS = self.fdecompose_shear_d2q9(fneq) * rho / 4.0
        else:
            deltaS = self.fdecompose_shear_d3q27(fneq) * rho
        deltaH = fneq - deltaS
        invBeta = 1.0 / beta
        gamma = invBeta - (2.0 - invBeta) * self.entropic_scalar_product(deltaS, deltaH, feq) / (
            tiny + self.entropic_scalar_product(deltaH, deltaH, feq)
        )

        fout = f - beta * (2.0 * deltaS + gamma[..., None] * deltaH)

        # add external force
        if self.force is not None:
            fout = self.apply_force(fout, feq, rho, u)
        return self.precision_policy.cast_to_output(fout)

    _KBC_METHODS = ("fdecompose_shear_d2q9", "fdecompose_shear_d3q27", "entropic_scalar_product")

    def _collision_applies_bounceback(self):
        """The per-direction collision applies the full-way bounce-back itself (see LBMBase.apply_fused_bounceback)."""
        return type(self).collision is KBCSim.collision and self._uses_default_kbc_methods()

    def _uses_default_kbc_methods(self):
        """Whether the macroscopic fields, equilibrium, forcing, momentum flux and the KBC decomposition are the defaults
        (then the per-direction collision applies)."""
        cls = type(self)
        return (
            cls.update_macroscopic is LBMBase.update_macroscopic
            and cls.equilibrium is LBMBase.equilibrium
            and _is_default_method(self, "apply_force", LBMBase.apply_force)
            and cls.momentum_flux is LBMBase.momentum_flux
            and all(getattr(cls, name) is getattr(KBCSim, name) for name in self._KBC_METHODS)
        )

    def _collision_components(self, f):
        """
        The KBC collision evaluated per lattice direction, for the default methods: the same equilibrium, shear decomposition
        (from the momentum flux of fneq), entropic stabilizer gamma and exact-difference forcing as the array path, with every
        (..., q) intermediate kept as q separate fields and the momentum flux as sparse sums with the constant c_a c_b
        coefficients. XLA fuses the whole collision into one kernel with one thread per node. Results can differ from the array
        path at roundoff level.
        """
        dtype = self.precision_policy.compute_dtype
        tiny = 1e-32
        beta = self.omega * 0.5
        invBeta = 1.0 / beta
        directions, weights, channels, rho, velocity = _per_direction_moments(self.lattice, f)
        usqr = 1.5 * sum(v * v for v in velocity)
        feq = []
        for direction, w in zip(directions, weights, strict=True):
            cu = 3.0 * _lattice_dot(direction, velocity)
            feq.append(rho * w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr))
        fneq = Components([fi - fe for fi, fe in zip(channels, feq, strict=True)], dtype)
        with component_mode():
            Pi = matvec_components(fneq, self.sparse_collision_terms(np.asarray(self.lattice.cc)), dtype)
            if self.dim == 2:
                shear = self._shear_from_momentum_flux_d2q9(Pi, fneq)
            else:
                shear = self._shear_from_momentum_flux_d3q27(Pi, fneq)
        deltaS = [s * rho / 4.0 for s in shear] if self.dim == 2 else [s * rho for s in shear]
        deltaH = [fn - ds for fn, ds in zip(fneq, deltaS, strict=True)]
        numerator = sum(ds * dh / fe for ds, dh, fe in zip(deltaS, deltaH, feq, strict=True))
        denominator = tiny + sum(dh * dh / fe for dh, fe in zip(deltaH, feq, strict=True))
        gamma = invBeta - (2.0 - invBeta) * numerator / denominator
        du, delta_usqr = _per_direction_force(self.get_force_field(rho[..., None]), self.dim, velocity) if self.force is not None else (None, None)
        outputs = []
        for direction, w, fi, ds, dh in zip(directions, weights, channels, deltaS, deltaH, strict=True):
            out = fi - beta * (2.0 * ds + gamma * dh)
            if du is not None:
                cu = 3.0 * _lattice_dot(direction, velocity)
                dcu = 3.0 * _lattice_dot(direction, du)
                out = out + rho * w * (dcu * (1.0 + cu + 0.5 * dcu) - delta_usqr)
            outputs.append(out)
        outputs = self.apply_fused_bounceback([jnp.broadcast_to(out, rho.shape) for out in outputs], channels, self.bounceback_mask)
        return self.stack_populations([self.precision_policy.cast_to_output(out) for out in outputs])

    @partial(jit, static_argnums=(0,), inline=True)
    def entropic_scalar_product(self, x, y, feq):
        """
        Compute the entropic scalar product of x and y to approximate gamma in KBC.

        Returns
        -------
        jax.numpy.array
            Entropic scalar product of x, y, and feq.
        """
        return jnp.sum(x * y / feq, axis=-1)

    @partial(jit, static_argnums=(0,), inline=True)
    def fdecompose_shear_d2q9(self, fneq):
        """
        Decompose fneq into shear components for D2Q9 lattice.

        Parameters
        ----------
        fneq (jax.numpy.array): Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.array
            Shear components of fneq.
        """
        return self._shear_from_momentum_flux_d2q9(self.momentum_flux(fneq), fneq)

    def _shear_from_momentum_flux_d2q9(self, Pi, like):
        """Shear part of the D2Q9 non-equilibrium populations from their momentum flux Pi (see fdecompose_shear_d2q9)."""
        # s_i = N (c_ix^2 - c_iy^2) + Pi_xy c_ix c_iy with N = Pi_xx - Pi_yy (Pi ordered xx, xy, yy), built from the
        # lattice vectors so that it holds for this lattice's direction ordering (with the later factor 1/4 it reproduces
        # the deviatoric part of Pi: N and Pi_xy)
        N = Pi[..., 0] - Pi[..., 2]
        c = np.asarray(self.lattice.c)
        s = {}
        for i in range(self.lattice.q):
            cx, cy = int(c[0, i]), int(c[1, i])
            if cx * cx != cy * cy:
                s[i] = N if cx * cx > cy * cy else -N
            elif cx * cy != 0:
                s[i] = Pi[..., 1] if cx * cy > 0 else -Pi[..., 1]
        s = stack_components(s, like)

        return s

    @partial(jit, static_argnums=(0,), inline=True)
    def fdecompose_shear_d3q27(self, fneq):
        """
        Decompose fneq into shear components for D3Q27 lattice.

        Parameters
        ----------
        fneq (jax.numpy.ndarray): Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.ndarray
            Shear components of fneq.
        """
        # if self.grid.dim == 3:
        #     diagonal    = (0, 3, 5)
        #     offdiagonal = (1, 2, 4)
        # elif self.grid.dim == 2:
        #     diagonal    = (0, 2)
        #     offdiagonal = (1,)

        # c=
        # array([[0, 0, 0],-----0
        #        [0, 0, -1],----1
        #        [0, 0, 1],-----2
        #        [0, -1, 0],----3
        #        [0, -1, -1],---4
        #        [0, -1, 1],----5
        #        [0, 1, 0],-----6
        #        [0, 1, -1],----7
        #        [0, 1, 1],-----8
        #        [-1, 0, 0],----9
        #        [-1, 0, -1],--10
        #        [-1, 0, 1],---11
        #        [-1, -1, 0],--12
        #        [-1, -1, -1],-13
        #        [-1, -1, 1],--14
        #        [-1, 1, 0],---15
        #        [-1, 1, -1],--16
        #        [-1, 1, 1],---17
        #        [1, 0, 0],----18
        #        [1, 0, -1],---19
        #        [1, 0, 1],----20
        #        [1, -1, 0],---21
        #        [1, -1, -1],--22
        #        [1, -1, 1],---23
        #        [1, 1, 0],----24
        #        [1, 1, -1],---25
        #        [1, 1, 1]])---26
        return self._shear_from_momentum_flux_d3q27(self.momentum_flux(fneq), fneq)

    def _shear_from_momentum_flux_d3q27(self, Pi, like):
        """Shear part of the D3Q27 non-equilibrium populations from their momentum flux Pi (see fdecompose_shear_d3q27)."""
        Nxz = Pi[..., 0] - Pi[..., 5]
        Nyz = Pi[..., 3] - Pi[..., 5]

        # For c = (i, 0, 0), c = (0, j, 0) and c = (0, 0, k)
        s = {}
        s[9] = (2.0 * Nxz - Nyz) / 6.0
        s[18] = (2.0 * Nxz - Nyz) / 6.0
        s[3] = (-Nxz + 2.0 * Nyz) / 6.0
        s[6] = (-Nxz + 2.0 * Nyz) / 6.0
        s[1] = (-Nxz - Nyz) / 6.0
        s[2] = (-Nxz - Nyz) / 6.0

        # For c = (i, j, 0)
        s[12] = Pi[..., 1] / 4.0
        s[24] = Pi[..., 1] / 4.0
        s[21] = -Pi[..., 1] / 4.0
        s[15] = -Pi[..., 1] / 4.0

        # For c = (i, 0, k)
        s[10] = Pi[..., 2] / 4.0
        s[20] = Pi[..., 2] / 4.0
        s[19] = -Pi[..., 2] / 4.0
        s[11] = -Pi[..., 2] / 4.0

        # For c = (0, j, k)
        s[8] = Pi[..., 4] / 4.0
        s[4] = Pi[..., 4] / 4.0
        s[7] = -Pi[..., 4] / 4.0
        s[5] = -Pi[..., 4] / 4.0
        s = stack_components(s, like)

        return s


class AdvectionDiffusionBGK(LBMBase):
    """
    Advection Diffusion Model based on the BGK model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vel = kwargs.get("vel", None)
        if self.vel is None:
            raise ValueError("Velocity must be specified for AdvectionDiffusionBGK.")

    def _collision_applies_bounceback(self):
        """The per-direction collision applies the full-way bounce-back itself (see LBMBase.apply_fused_bounceback)."""
        return type(self).collision is AdvectionDiffusionBGK.collision and type(self).equilibrium is LBMBase.equilibrium

    @partial(jit, static_argnums=(0,))
    def collision(self, f):
        """
        BGK collision step for lattice.
        """
        f = self.precision_policy.cast_to_compute(f)
        if type(self).equilibrium is LBMBase.equilibrium:
            # Per-direction form (one kernel per node, see BGKSim._collision_components); same formulas
            directions, weights = _direction_tables(self.lattice)
            channels = [f[..., i] for i in range(self.q)]
            rho = sum(channels)
            velocity = [self.vel[..., axis] for axis in range(self.dim)]
            usqr = 1.5 * sum(v * v for v in velocity)
            outputs = []
            for direction, w, fi in zip(directions, weights, channels, strict=True):
                cu = 3.0 * _lattice_dot(direction, velocity)
                feq = rho * w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr)
                outputs.append(fi - self.omega * (fi - feq))
            outputs = self.apply_fused_bounceback(outputs, channels, self.bounceback_mask)
            return self.stack_populations([self.precision_policy.cast_to_output(out) for out in outputs])
        rho = jnp.sum(f, axis=-1, keepdims=True)
        feq = self.equilibrium(rho, self.vel, cast_output=False)
        fneq = f - feq
        fout = f - self.omega * fneq
        return self.precision_policy.cast_to_output(fout)


class MRTSim(LBMBase):
    """
    Multi-relaxation time model.
    """

    # Direction-major populations in run() by default (with the per-direction collision: +26-31% in 3D, +4% in 2D;
    # see LBMBase._direction_major_step)
    _direction_major_dims = (2, 3)

    def __init__(self, **kwargs):
        kwargs.update({"omega": 1.0})
        super().__init__(**kwargs)
        self.s_rho = kwargs.get("s_rho")
        self.s_e = kwargs.get("s_e")
        self.s_eta = kwargs.get("s_eta")
        self.s_j = kwargs.get("s_j")
        self.s_q = kwargs.get("s_q")
        self.s_v = self.omega
        self.M_inv = jnp.array(
            np.transpose(np.linalg.inv(kwargs.get("M"))),
            dtype=self.precision_policy.compute_dtype,
        )
        self.M = jnp.array(np.transpose(kwargs.get("M")), dtype=self.precision_policy.compute_dtype)
        if isinstance(self.lattice, LatticeD2Q9):
            self.S = jnp.array(
                np.diag([self.s_rho, self.s_e, self.s_eta, self.s_j, self.s_q, self.s_j, self.s_q, self.s_v, self.s_v]),
                dtype=self.precision_policy.compute_dtype,
            )
        elif isinstance(self.lattice, LatticeD3Q19):
            self.s_pi = kwargs.get("s_pi")
            self.s_m = kwargs.get("s_m")
            self.S = jnp.array(
                np.diag([
                    self.s_rho,
                    self.s_e,
                    self.s_eta,
                    self.s_j,
                    self.s_q,
                    self.s_j,
                    self.s_q,
                    self.s_j,
                    self.s_q,
                    self.s_v,
                    self.s_pi,
                    self.s_v,
                    self.s_pi,
                    self.s_v,
                    self.s_v,
                    self.s_v,
                    self.s_m,
                    self.s_m,
                    self.s_m,
                ]),
                dtype=self.precision_policy.compute_dtype,
            )
        else:
            NotImplementedError(f"Lattice type {self.lattice.name} has not been implemented")

        # Fused collision matrix K = M @ S @ M_inv, replacing the three separate moment-space matrix multiplies
        # (f @ M, relax, @ M_inv) in collision() with symbolic, sparse-coefficient columns compiled from K's
        # nonzero entries - see collision() and MultiphaseMRT.collision (same fusion identity, without a
        # surface-tension term since this is the single-phase model).
        collision_matrix = np.asarray(jnp.dot(jnp.dot(self.M, self.S), self.M_inv))
        self.collision_matrix = jnp.array(collision_matrix, dtype=self.precision_policy.compute_dtype)
        # Coefficients in the compute precision (see LBMBase.sparse_collision_terms)
        self.collision_terms = self.sparse_collision_terms(collision_matrix)

    @partial(jit, static_argnums=(0,))
    def collision(self, f):
        """
        MRT collision step for lattice, using a symbolic (sparse-coefficient) fused collision matrix instead of
        three separate moment-space matrix multiplies. See MultiphaseMRT.collision for the fusion identity.
        """
        f = self.precision_policy.cast_to_compute(f)
        if self._uses_default_mrt_methods():
            return self._collision_components(f)
        rho, u = self.update_macroscopic(f)
        feq = self.equilibrium(rho, u)
        difference = f - feq

        outputs = []
        for output_direction, terms in enumerate(self.collision_terms):
            relaxed = sum(difference[..., input_direction] * coefficient for input_direction, coefficient in terms)
            outputs.append(f[..., output_direction] - relaxed)

        if self.force is not None:
            if _is_default_method(self, "apply_force", MRTSim.apply_force):
                delta_feq = self._compute_force_delta_feq(rho, u)
                outputs = [output + delta_feq[..., i] for i, output in enumerate(outputs)]
            else:
                # A user (or Thermal) apply_force takes moment-space quantities (m, meq), as documented in apply_force
                m = self.apply_force(jnp.dot(jnp.stack(outputs, axis=-1), self.M), jnp.dot(feq, self.M), rho, u)
                fout = jnp.dot(m, self.M_inv)
                outputs = [fout[..., i] for i in range(self.q)]
        outputs = self.apply_fused_bounceback(outputs, [f[..., i] for i in range(self.q)], self.bounceback_mask)
        return self.stack_populations([self.precision_policy.cast_to_output(output) for output in outputs])

    def _collision_applies_bounceback(self):
        """Both collision paths apply the full-way bounce-back themselves (see LBMBase.apply_fused_bounceback)."""
        return type(self).collision is MRTSim.collision

    def _uses_default_mrt_methods(self):
        """Whether the macroscopic fields, equilibrium and forcing are the defaults (then the per-direction collision applies)."""
        cls = type(self)
        return (
            cls.update_macroscopic is LBMBase.update_macroscopic
            and cls.equilibrium is LBMBase.equilibrium
            and cls._compute_force_delta_feq is MRTSim._compute_force_delta_feq
            and _is_default_method(self, "apply_force", MRTSim.apply_force)
        )

    def _collision_components(self, f):
        """
        The MRT collision with every term evaluated per lattice direction (moments, equilibrium, relaxation with the sparse
        fused matrix, exact-difference forcing), for the default methods. As in the array path, the equilibrium is rounded to
        the output precision before relaxation (equilibrium() is called with cast_output=True there). Results can differ from
        the array path at roundoff level (summation order of the moments).
        """
        directions, weights, channels, rho, velocity = _per_direction_moments(self.lattice, f)
        usqr = 1.5 * sum(v * v for v in velocity)
        difference = []
        for direction, w, fi in zip(directions, weights, channels, strict=True):
            cu = 3.0 * _lattice_dot(direction, velocity)
            feq = self.precision_policy.cast_to_output(rho * w * (1.0 + cu * (1.0 + 0.5 * cu) - usqr))
            difference.append(fi - feq)
        outputs = []
        for output_direction, terms in enumerate(self.collision_terms):
            relaxed = sum(difference[input_direction] * coefficient for input_direction, coefficient in terms)
            outputs.append(channels[output_direction] - relaxed)
        du, delta_usqr = _per_direction_force(self.get_force_field(rho[..., None]), self.dim, velocity) if self.force is not None else (None, None)
        if du is not None:
            for i, (direction, w) in enumerate(zip(directions, weights, strict=True)):
                cu = 3.0 * _lattice_dot(direction, velocity)
                dcu = 3.0 * _lattice_dot(direction, du)
                outputs[i] = outputs[i] + rho * w * (dcu * (1.0 + cu + 0.5 * dcu) - delta_usqr)
        outputs = self.apply_fused_bounceback(outputs, channels, self.bounceback_mask)
        return self.stack_populations([self.precision_policy.cast_to_output(output) for output in outputs])

    @partial(jit, static_argnums=(0,), inline=True)
    def _compute_force_delta_feq(self, rho, u):
        """
        Real-space compact EDM difference delta_feq = feq(rho, u + du) - feq(rho, u), computed directly from
        cu, dcu and delta_usqr instead of building a full feq_force array via equilibrium(). Shared by
        collision (added directly, real space) and apply_force (transformed into moment space with M, for
        callers using the unfused, moment-space collision path).

        Parameters
        ----------
        rho (jax.numpy.ndarray): Density field.

        u (jax.numpy.ndarray): Velocity field.

        Returns
        -------
        (jax.numpy.ndarray): Real-space delta_feq.
        """
        du = self.get_force_field(rho)
        c = jnp.array(self.c, dtype=self.precision_policy.compute_dtype)
        cu = 3.0 * jnp.dot(u, c)
        dcu = 3.0 * jnp.dot(du, c)
        delta_usqr = 1.5 * (2.0 * jnp.sum(u * du, axis=-1, keepdims=True) + jnp.sum(jnp.square(du), axis=-1, keepdims=True))
        return rho * self.w * (dcu * (1.0 + cu + 0.5 * dcu) - delta_usqr)

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, m, meq, rho, u):
        """
        add force based on exact-difference method due to Kupershtokh, in moment space.

        LBMBase.apply_force subtracts feq in population space, which does not apply here since m/meq are
        moment-space quantities. Instead, the real-space delta_feq (see _compute_force_delta_feq) is
        transformed into moment space with M: since M is linear, dot(delta_feq, M) ==
        dot(feq(rho, u + du), M) - dot(feq(rho, u), M), an exact substitute for subtracting a moment-space
        equilibrium from a population-space one. Provided for callers using the unfused, moment-space collision
        path (e.g. Thermal.apply_force_thermal, when the wrapped fluid_solver is an MRTSim). collision() adds
        delta_feq directly in real space for this default, and calls apply_force with the post-collision and equilibrium
        moments only when it is overridden (by a subclass or on the instance).

        Parameters
        ----------
        m (jax.numpy.ndarray): Post-collision moments.

        meq (jax.numpy.ndarray): Equilibrium moments. Unused - kept for interface compatibility, since the
            compact difference formula only needs rho, u and the force.

        rho (jax.numpy.ndarray): Density field.

        u (jax.numpy.ndarray): Velocity field.

        Returns
        -------
        (jax.numpy.ndarray): Post-collision moments with the force applied.

        References
        ----------
        1. Kupershtokh, A. (2004). New method of incorporating a body force term into the lattice Boltzmann
        equation. In Proceedings of the 5th International EHD Workshop (pp. 241-246). University of Poitiers.
        """
        delta_feq = self._compute_force_delta_feq(rho, u)
        return m + jnp.dot(delta_feq, self.M)


class CLBMSim(LBMBase):
    """
    Central moment (cascaded) collision model.

    The second-order central moments relax with s_2 (diagonal, normal-stress differences), s_b (bulk) and s_v (off-diagonal
    shear). s_v defaults to s_2, which gives an isotropic viscosity nu = cs^2 (1 / s_2 - 1/2); omega is set to s_v.
    """

    def __init__(self, **kwargs):
        s_v = kwargs.get("s_v")
        kwargs["omega"] = kwargs.get("s_2") if s_v is None else s_v
        super().__init__(**kwargs)
        self.M_inv = jnp.array(
            np.transpose(np.linalg.inv(kwargs.get("M"))),
            dtype=self.precision_policy.compute_dtype,
        )
        self.M = jnp.array(np.transpose(kwargs.get("M")), dtype=self.precision_policy.compute_dtype)
        self.s_0 = kwargs.get("s_0")
        self.s_1 = kwargs.get("s_1")
        self.s_b = kwargs.get("s_b")
        self.s_2 = kwargs.get("s_2")
        self.s_3 = kwargs.get("s_3")
        self.s_4 = kwargs.get("s_4")
        self.s_v = self.omega
        if isinstance(self.lattice, LatticeD2Q9):
            self.S = jnp.array(
                np.diag([self.s_0, self.s_1, self.s_1, self.s_b, self.s_2, self.s_2, self.s_3, self.s_3, self.s_4]),
                dtype=self.precision_policy.compute_dtype,
            )
        elif isinstance(self.lattice, LatticeD3Q19):
            self.s_plus = (self.s_b + 2 * self.s_2) / 3
            self.s_minus = (self.s_b - self.s_2) / 3

            S = np.diag([
                self.s_0,
                self.s_1,
                self.s_1,
                self.s_1,
                self.s_v,
                self.s_v,
                self.s_v,
                self.s_plus,
                self.s_plus,
                self.s_plus,
                self.s_3,
                self.s_3,
                self.s_3,
                self.s_3,
                self.s_3,
                self.s_3,
                self.s_4,
                self.s_4,
                self.s_4,
            ])
            S[7, 8] = self.s_minus
            S[7, 9] = self.s_minus
            S[8, 7] = self.s_minus
            S[8, 9] = self.s_minus
            S[9, 7] = self.s_minus
            S[9, 8] = self.s_minus
            self.S = jnp.array(S, dtype=self.precision_policy.compute_dtype)

        elif isinstance(self.lattice, LatticeD3Q27):
            self.s_plus = (self.s_b + 2 * self.s_2) / 3
            self.s_minus = (self.s_b - self.s_2) / 3
            self.s_3b = kwargs.get("s_3b")
            self.s_4b = kwargs.get("s_4b")
            self.s_5 = kwargs.get("s_5")
            self.s_6 = kwargs.get("s_6")

            S = np.diag([
                self.s_0,
                self.s_1,
                self.s_1,
                self.s_1,
                self.s_v,
                self.s_v,
                self.s_v,
                self.s_plus,
                self.s_plus,
                self.s_plus,
                self.s_3,
                self.s_3,
                self.s_3,
                self.s_3,
                self.s_3,
                self.s_3,
                self.s_3b,
                self.s_4,
                self.s_4,
                self.s_4,
                self.s_4b,
                self.s_4b,
                self.s_4b,
                self.s_5,
                self.s_5,
                self.s_5,
                self.s_6,
            ])
            S[7, 8] = self.s_minus
            S[7, 9] = self.s_minus
            S[8, 7] = self.s_minus
            S[8, 9] = self.s_minus
            S[9, 7] = self.s_minus
            S[9, 8] = self.s_minus
            self.S = jnp.array(S, dtype=self.precision_policy.compute_dtype)

    @partial(jit, static_argnums=(0,), inline=True)
    def macroscopic_velocity(self, f, rho):
        """
        macroscopic_velocity computes the velocity and incorporates forces into velocity for Exact Difference Method (EDM) (used for SRT and MRT collision) models
        and the consistent forcing scheme developed by LinLin Fei et. al (for Cascaded LBM). This is used for post-processing only and not for equilibrium distribution computation.

        Parameters
        ----------
        f (jax.numpy.ndarray): Distribution arrays.

        rho (jax.numpy.ndarray): Density fields.

        Returns
        -------
        u: jax.numpy.ndarray
            Velocity fields.
        """
        u = lattice_density_momentum(self.lattice.c, self.precision_policy.cast_to_compute(f))[1] / rho
        if self.force is not None:
            return u + 0.5 * self.force / rho
        else:
            return u

    @partial(jit, static_argnums=(0,))
    def compute_central_moment(self, m, u):
        if isinstance(self.lattice, LatticeD2Q9):

            def shift(m, u):
                ux = u[..., 0]
                uy = u[..., 1]
                usq = ux**2 + uy**2
                udiff = ux**2 - uy**2
                T = {}
                T[0] = m[..., 0]
                T[1] = -ux * m[..., 0] + m[..., 1]
                T[2] = -uy * m[..., 0] + m[..., 2]
                T[3] = usq * m[..., 0] - 2 * ux * m[..., 1] - 2 * uy * m[..., 2] + m[..., 3]
                T[4] = udiff * m[..., 0] - 2 * ux * m[..., 1] + 2 * uy * m[..., 2] + m[..., 4]
                T[5] = ux * uy * m[..., 0] - uy * m[..., 1] - ux * m[..., 2] + m[..., 5]
                T[6] = (
                    -(ux**2) * uy * m[..., 0]
                    + 2 * ux * uy * m[..., 1]
                    + ux**2 * m[..., 2]
                    - 0.5 * uy * m[..., 3]
                    - 0.5 * uy * m[..., 4]
                    - 2 * ux * m[..., 5]
                    + m[..., 6]
                )
                T[7] = (
                    -(uy**2) * ux * m[..., 0]
                    + uy**2 * m[..., 1]
                    + 2 * ux * uy * m[..., 2]
                    - 0.5 * ux * m[..., 3]
                    + 0.5 * ux * m[..., 4]
                    - 2 * uy * m[..., 5]
                    + m[..., 7]
                )
                T[8] = (
                    (uy**2 * ux**2) * m[..., 0]
                    - 2 * ux * uy**2 * m[..., 1]
                    - 2 * uy * ux**2 * m[..., 2]
                    + 0.5 * usq * m[..., 3]
                    - 0.5 * udiff * m[..., 4]
                    + 4 * ux * uy * m[..., 5]
                    - 2 * uy * m[..., 6]
                    - 2 * ux * m[..., 7]
                    + m[..., 8]
                )
                T = stack_components(T, m)
                return T

            return shift(m, u)

        elif isinstance(self.lattice, LatticeD3Q19):

            def shift(m, u):
                ux = u[..., 0]
                uy = u[..., 1]
                uz = u[..., 2]
                T = {}
                T[0] = m[..., 0]
                T[1] = -ux * m[..., 0] + m[..., 1]
                T[2] = -uy * m[..., 0] + m[..., 2]
                T[3] = -uz * m[..., 0] + m[..., 3]
                T[4] = ux * uy * m[..., 0] - uy * m[..., 1] - ux * m[..., 2] + m[..., 4]
                T[5] = ux * uz * m[..., 0] - uz * m[..., 1] - ux * m[..., 3] + m[..., 5]
                T[6] = uy * uz * m[..., 0] - uz * m[..., 2] - uy * m[..., 3] + m[..., 6]
                T[7] = (ux**2) * m[..., 0] - 2 * ux * m[..., 1] + m[..., 7]
                T[8] = (uy**2) * m[..., 0] - 2 * uy * m[..., 2] + m[..., 8]
                T[9] = (uz**2) * m[..., 0] - 2 * uz * m[..., 3] + m[..., 9]
                T[10] = -ux * (uy**2) * m[..., 0] + (uy**2) * m[..., 1] + 2 * ux * uy * m[..., 2] - 2 * uy * m[..., 4] - ux * m[..., 8] + m[..., 10]
                T[11] = -ux * (uz**2) * m[..., 0] + (uz**2) * m[..., 1] + 2 * ux * uz * m[..., 3] - 2 * uz * m[..., 5] - ux * m[..., 9] + m[..., 11]
                T[12] = -(ux**2) * uy * m[..., 0] + 2 * ux * uy * m[..., 1] + (ux**2) * m[..., 2] - 2 * ux * m[..., 4] - uy * m[..., 7] + m[..., 12]
                T[13] = -(ux**2) * uz * m[..., 0] + 2 * ux * uz * m[..., 1] + (ux**2) * m[..., 3] - 2 * ux * m[..., 5] - uz * m[..., 7] + m[..., 13]
                T[14] = -uy * (uz**2) * m[..., 0] + (uz**2) * m[..., 2] + 2 * uy * uz * m[..., 3] - 2 * uz * m[..., 6] - uy * m[..., 9] + m[..., 14]
                T[15] = -(uy**2) * uz * m[..., 0] + 2 * uy * uz * m[..., 2] + (uy**2) * m[..., 3] - 2 * uy * m[..., 6] - uz * m[..., 8] + m[..., 15]
                T[16] = (
                    (ux**2) * (uy**2) * m[..., 0]
                    - 2 * ux * (uy**2) * m[..., 1]
                    - 2 * uy * (ux**2) * m[..., 2]
                    + 4 * ux * uy * m[..., 4]
                    + (uy**2) * m[..., 7]
                    + (ux**2) * m[..., 8]
                    - 2 * ux * m[..., 10]
                    - 2 * uy * m[..., 12]
                    + m[..., 16]
                )
                T[17] = (
                    (ux**2) * (uz**2) * m[..., 0]
                    - 2 * ux * (uz**2) * m[..., 1]
                    - 2 * uz * (ux**2) * m[..., 3]
                    + 4 * ux * uz * m[..., 5]
                    + (uz**2) * m[..., 7]
                    + (ux**2) * m[..., 9]
                    - 2 * ux * m[..., 11]
                    - 2 * uz * m[..., 13]
                    + m[..., 17]
                )
                T[18] = (
                    (uy**2) * (uz**2) * m[..., 0]
                    - 2 * uy * (uz**2) * m[..., 2]
                    - 2 * uz * (uy**2) * m[..., 3]
                    + 4 * uy * uz * m[..., 6]
                    + (uz**2) * m[..., 8]
                    + (uy**2) * m[..., 9]
                    - 2 * uy * m[..., 14]
                    - 2 * uz * m[..., 15]
                    + m[..., 18]
                )
                T = stack_components(T, m)
                return T

            return shift(m, u)

        elif isinstance(self.lattice, LatticeD3Q27):

            def shift(m, u):
                ux = u[..., 0]
                uy = u[..., 1]
                uz = u[..., 2]
                T = {}
                T[0] = m[..., 0]
                T[1] = m[..., 1] - m[..., 0] * ux
                T[2] = m[..., 2] - m[..., 0] * uy
                T[3] = m[..., 3] - m[..., 0] * uz
                T[4] = m[..., 4] - m[..., 2] * ux - m[..., 1] * uy + m[..., 0] * ux * uy
                T[5] = m[..., 5] - m[..., 3] * ux - m[..., 1] * uz + m[..., 0] * ux * uz
                T[6] = m[..., 6] - m[..., 3] * uy - m[..., 2] * uz + m[..., 0] * uy * uz
                T[7] = m[..., 0] * ux**2 - 2 * m[..., 1] * ux + m[..., 7]
                T[8] = m[..., 0] * uy**2 - 2 * m[..., 2] * uy + m[..., 8]
                T[9] = m[..., 0] * uz**2 - 2 * m[..., 3] * uz + m[..., 9]
                T[10] = m[..., 10] - m[..., 8] * ux - 2 * m[..., 4] * uy + m[..., 1] * uy**2 - m[..., 0] * ux * uy**2 + 2 * m[..., 2] * ux * uy
                T[11] = m[..., 11] - m[..., 9] * ux - 2 * m[..., 5] * uz + m[..., 1] * uz**2 - m[..., 0] * ux * uz**2 + 2 * m[..., 3] * ux * uz
                T[12] = m[..., 12] - 2 * m[..., 4] * ux - m[..., 7] * uy + m[..., 2] * ux**2 - m[..., 0] * ux**2 * uy + 2 * m[..., 1] * ux * uy
                T[13] = m[..., 13] - 2 * m[..., 5] * ux - m[..., 7] * uz + m[..., 3] * ux**2 - m[..., 0] * ux**2 * uz + 2 * m[..., 1] * ux * uz
                T[14] = m[..., 14] - m[..., 9] * uy - 2 * m[..., 6] * uz + m[..., 2] * uz**2 - m[..., 0] * uy * uz**2 + 2 * m[..., 3] * uy * uz
                T[15] = m[..., 15] - 2 * m[..., 6] * uy - m[..., 8] * uz + m[..., 3] * uy**2 - m[..., 0] * uy**2 * uz + 2 * m[..., 2] * uy * uz
                T[16] = (
                    m[..., 16]
                    - m[..., 6] * ux
                    - m[..., 5] * uy
                    - m[..., 4] * uz
                    + m[..., 3] * ux * uy
                    + m[..., 2] * ux * uz
                    + m[..., 1] * uy * uz
                    - m[..., 0] * ux * uy * uz
                )
                T[17] = (
                    m[..., 0] * ux**2 * uy**2
                    - 2 * m[..., 2] * ux**2 * uy
                    + m[..., 8] * ux**2
                    - 2 * m[..., 1] * ux * uy**2
                    + 4 * m[..., 4] * ux * uy
                    - 2 * m[..., 10] * ux
                    + m[..., 7] * uy**2
                    - 2 * m[..., 12] * uy
                    + m[..., 17]
                )
                T[18] = (
                    m[..., 0] * ux**2 * uz**2
                    - 2 * m[..., 3] * ux**2 * uz
                    + m[..., 9] * ux**2
                    - 2 * m[..., 1] * ux * uz**2
                    + 4 * m[..., 5] * ux * uz
                    - 2 * m[..., 11] * ux
                    + m[..., 7] * uz**2
                    - 2 * m[..., 13] * uz
                    + m[..., 18]
                )
                T[19] = (
                    m[..., 0] * uy**2 * uz**2
                    - 2 * m[..., 3] * uy**2 * uz
                    + m[..., 9] * uy**2
                    - 2 * m[..., 2] * uy * uz**2
                    + 4 * m[..., 6] * uy * uz
                    - 2 * m[..., 14] * uy
                    + m[..., 8] * uz**2
                    - 2 * m[..., 15] * uz
                    + m[..., 19]
                )
                T[20] = (
                    m[..., 20]
                    - 2 * m[..., 16] * ux
                    - m[..., 13] * uy
                    - m[..., 12] * uz
                    + m[..., 6] * ux**2
                    - m[..., 3] * ux**2 * uy
                    - m[..., 2] * ux**2 * uz
                    + 2 * m[..., 5] * ux * uy
                    + 2 * m[..., 4] * ux * uz
                    + m[..., 7] * uy * uz
                    - 2 * m[..., 1] * ux * uy * uz
                    + m[..., 0] * ux**2 * uy * uz
                )
                T[21] = (
                    m[..., 21]
                    - m[..., 15] * ux
                    - 2 * m[..., 16] * uy
                    - m[..., 10] * uz
                    + m[..., 5] * uy**2
                    - m[..., 3] * ux * uy**2
                    - m[..., 1] * uy**2 * uz
                    + 2 * m[..., 6] * ux * uy
                    + m[..., 8] * ux * uz
                    + 2 * m[..., 4] * uy * uz
                    - 2 * m[..., 2] * ux * uy * uz
                    + m[..., 0] * ux * uy**2 * uz
                )
                T[22] = (
                    m[..., 22]
                    - m[..., 14] * ux
                    - m[..., 11] * uy
                    - 2 * m[..., 16] * uz
                    + m[..., 4] * uz**2
                    - m[..., 2] * ux * uz**2
                    - m[..., 1] * uy * uz**2
                    + m[..., 9] * ux * uy
                    + 2 * m[..., 6] * ux * uz
                    + 2 * m[..., 5] * uy * uz
                    - 2 * m[..., 3] * ux * uy * uz
                    + m[..., 0] * ux * uy * uz**2
                )
                T[23] = (
                    m[..., 23]
                    - m[..., 19] * ux
                    - 2 * m[..., 22] * uy
                    - 2 * m[..., 21] * uz
                    + m[..., 11] * uy**2
                    + m[..., 10] * uz**2
                    - m[..., 9] * ux * uy**2
                    - m[..., 8] * ux * uz**2
                    - 2 * m[..., 4] * uy * uz**2
                    - 2 * m[..., 5] * uy**2 * uz
                    + m[..., 1] * uy**2 * uz**2
                    + 2 * m[..., 14] * ux * uy
                    + 2 * m[..., 15] * ux * uz
                    + 4 * m[..., 16] * uy * uz
                    - 4 * m[..., 6] * ux * uy * uz
                    + 2 * m[..., 2] * ux * uy * uz**2
                    + 2 * m[..., 3] * ux * uy**2 * uz
                    - m[..., 0] * ux * uy**2 * uz**2
                )
                T[24] = (
                    m[..., 24]
                    - 2 * m[..., 22] * ux
                    - m[..., 18] * uy
                    - 2 * m[..., 20] * uz
                    + m[..., 14] * ux**2
                    + m[..., 12] * uz**2
                    - m[..., 9] * ux**2 * uy
                    - 2 * m[..., 4] * ux * uz**2
                    - 2 * m[..., 6] * ux**2 * uz
                    - m[..., 7] * uy * uz**2
                    + m[..., 2] * ux**2 * uz**2
                    + 2 * m[..., 11] * ux * uy
                    + 4 * m[..., 16] * ux * uz
                    + 2 * m[..., 13] * uy * uz
                    - 4 * m[..., 5] * ux * uy * uz
                    + 2 * m[..., 1] * ux * uy * uz**2
                    + 2 * m[..., 3] * ux**2 * uy * uz
                    - m[..., 0] * ux**2 * uy * uz**2
                )
                T[25] = (
                    m[..., 25]
                    - 2 * m[..., 21] * ux
                    - 2 * m[..., 20] * uy
                    - m[..., 17] * uz
                    + m[..., 15] * ux**2
                    + m[..., 13] * uy**2
                    - 2 * m[..., 5] * ux * uy**2
                    - 2 * m[..., 6] * ux**2 * uy
                    - m[..., 8] * ux**2 * uz
                    - m[..., 7] * uy**2 * uz
                    + m[..., 3] * ux**2 * uy**2
                    + 4 * m[..., 16] * ux * uy
                    + 2 * m[..., 10] * ux * uz
                    + 2 * m[..., 12] * uy * uz
                    - 4 * m[..., 4] * ux * uy * uz
                    + 2 * m[..., 1] * ux * uy**2 * uz
                    + 2 * m[..., 2] * ux**2 * uy * uz
                    - m[..., 0] * ux**2 * uy**2 * uz
                )
                T[26] = (
                    m[..., 0] * ux**2 * uy**2 * uz**2
                    - 2 * m[..., 3] * ux**2 * uy**2 * uz
                    + m[..., 9] * ux**2 * uy**2
                    - 2 * m[..., 2] * ux**2 * uy * uz**2
                    + 4 * m[..., 6] * ux**2 * uy * uz
                    - 2 * m[..., 14] * ux**2 * uy
                    + m[..., 8] * ux**2 * uz**2
                    - 2 * m[..., 15] * ux**2 * uz
                    + m[..., 19] * ux**2
                    - 2 * m[..., 1] * ux * uy**2 * uz**2
                    + 4 * m[..., 5] * ux * uy**2 * uz
                    - 2 * m[..., 11] * ux * uy**2
                    + 4 * m[..., 4] * ux * uy * uz**2
                    - 8 * m[..., 16] * ux * uy * uz
                    + 4 * m[..., 22] * ux * uy
                    - 2 * m[..., 10] * ux * uz**2
                    + 4 * m[..., 21] * ux * uz
                    - 2 * m[..., 23] * ux
                    + m[..., 7] * uy**2 * uz**2
                    - 2 * m[..., 13] * uy**2 * uz
                    + m[..., 18] * uy**2
                    - 2 * m[..., 12] * uy * uz**2
                    + 4 * m[..., 20] * uy * uz
                    - 2 * m[..., 24] * uy
                    + m[..., 17] * uz**2
                    - 2 * m[..., 25] * uz
                    + m[..., 26]
                )
                T = stack_components(T, m)

                return T

            return shift(m, u)

    @partial(jit, static_argnums=(0,))
    def compute_central_moment_inverse(self, T, u):
        if isinstance(self.lattice, LatticeD2Q9):

            def shift_inverse(T, u):
                ux = u[..., 0]
                uy = u[..., 1]
                usq = ux**2 + uy**2
                udiff = ux**2 - uy**2
                m = {}
                m[0] = T[..., 0]
                m[1] = ux * T[..., 0] + T[..., 1]
                m[2] = uy * T[..., 0] + T[..., 2]
                m[3] = usq * T[..., 0] + 2 * ux * T[..., 1] + 2 * uy * T[..., 2] + T[..., 3]
                m[4] = udiff * T[..., 0] + 2 * ux * T[..., 1] - 2 * uy * T[..., 2] + T[..., 4]
                m[5] = ux * uy * T[..., 0] + uy * T[..., 1] + ux * T[..., 2] + T[..., 5]
                m[6] = (
                    (ux**2) * uy * T[..., 0]
                    + 2 * ux * uy * T[..., 1]
                    + ux**2 * T[..., 2]
                    + 0.5 * uy * T[..., 3]
                    + 0.5 * uy * T[..., 4]
                    + 2 * ux * T[..., 5]
                    + T[..., 6]
                )
                m[7] = (
                    (uy**2) * ux * T[..., 0]
                    + uy**2 * T[..., 1]
                    + 2 * ux * uy * T[..., 2]
                    + 0.5 * ux * T[..., 3]
                    - 0.5 * ux * T[..., 4]
                    + 2 * uy * T[..., 5]
                    + T[..., 7]
                )
                m[8] = (
                    (uy**2 * ux**2) * T[..., 0]
                    + 2 * ux * uy**2 * T[..., 1]
                    + 2 * uy * ux**2 * T[..., 2]
                    + 0.5 * usq * T[..., 3]
                    - 0.5 * udiff * T[..., 4]
                    + 4 * ux * uy * T[..., 5]
                    + 2 * uy * T[..., 6]
                    + 2 * ux * T[..., 7]
                    + T[..., 8]
                )
                m = stack_components(m, T)
                return m

            return shift_inverse(T, u)

        elif isinstance(self.lattice, LatticeD3Q19):

            def shift_inverse(T, u):
                ux = u[..., 0]
                uy = u[..., 1]
                uz = u[..., 2]
                m = {}
                m[0] = T[..., 0]
                m[1] = ux * T[..., 0] + T[..., 1]
                m[2] = uy * T[..., 0] + T[..., 2]
                m[3] = uz * T[..., 0] + T[..., 3]
                m[4] = ux * uy * T[..., 0] + uy * T[..., 1] + ux * T[..., 2] + T[..., 4]
                m[5] = ux * uz * T[..., 0] + uz * T[..., 1] + ux * T[..., 3] + T[..., 5]
                m[6] = uy * uz * T[..., 0] + uz * T[..., 2] + uy * T[..., 3] + T[..., 6]
                m[7] = (ux**2) * T[..., 0] + 2 * ux * T[..., 1] + T[..., 7]
                m[8] = (uy**2) * T[..., 0] + 2 * uy * T[..., 2] + T[..., 8]
                m[9] = (uz**2) * T[..., 0] + 2 * uz * T[..., 3] + T[..., 9]
                m[10] = ux * (uy**2) * T[..., 0] + (uy**2) * T[..., 1] + 2 * ux * uy * T[..., 2] + 2 * uy * T[..., 4] + ux * T[..., 8] + T[..., 10]
                m[11] = ux * (uz**2) * T[..., 0] + (uz**2) * T[..., 1] + 2 * ux * uz * T[..., 3] + 2 * uz * T[..., 5] + ux * T[..., 9] + T[..., 11]
                m[12] = (ux**2) * uy * T[..., 0] + 2 * ux * uy * T[..., 1] + (ux**2) * T[..., 2] + 2 * ux * T[..., 4] + uy * T[..., 7] + T[..., 12]
                m[13] = (ux**2) * uz * T[..., 0] + 2 * ux * uz * T[..., 1] + (ux**2) * T[..., 3] + 2 * ux * T[..., 5] + uz * T[..., 7] + T[..., 13]
                m[14] = uy * (uz**2) * T[..., 0] + (uz**2) * T[..., 2] + 2 * uy * uz * T[..., 3] + 2 * uz * T[..., 6] + uy * T[..., 9] + T[..., 14]
                m[15] = (uy**2) * uz * T[..., 0] + 2 * uy * uz * T[..., 2] + (uy**2) * T[..., 3] + 2 * uy * T[..., 6] + uz * T[..., 8] + T[..., 15]
                m[16] = (
                    (ux**2) * (uy**2) * T[..., 0]
                    + 2 * ux * (uy**2) * T[..., 1]
                    + 2 * uy * (ux**2) * T[..., 2]
                    + 4 * ux * uy * T[..., 4]
                    + (uy**2) * T[..., 7]
                    + (ux**2) * T[..., 8]
                    + 2 * ux * T[..., 10]
                    + 2 * uy * T[..., 12]
                    + T[..., 16]
                )
                m[17] = (
                    (ux**2) * (uz**2) * T[..., 0]
                    + 2 * ux * (uz**2) * T[..., 1]
                    + 2 * uz * (ux**2) * T[..., 3]
                    + 4 * ux * uz * T[..., 5]
                    + (uz**2) * T[..., 7]
                    + (ux**2) * T[..., 9]
                    + 2 * ux * T[..., 11]
                    + 2 * uz * T[..., 13]
                    + T[..., 17]
                )
                m[18] = (
                    (uy**2) * (uz**2) * T[..., 0]
                    + 2 * uy * (uz**2) * T[..., 2]
                    + 2 * uz * (uy**2) * T[..., 3]
                    + 4 * uy * uz * T[..., 6]
                    + (uz**2) * T[..., 8]
                    + (uy**2) * T[..., 9]
                    + 2 * uy * T[..., 14]
                    + 2 * uz * T[..., 15]
                    + T[..., 18]
                )
                m = stack_components(m, T)
                return m

            return shift_inverse(T, u)

        elif isinstance(self.lattice, LatticeD3Q27):

            def shift_inverse(T, u):
                ux = u[..., 0]
                uy = u[..., 1]
                uz = u[..., 2]
                m = {}
                m[0] = T[..., 0]
                m[1] = T[..., 1] + T[..., 0] * ux
                m[2] = T[..., 2] + T[..., 0] * uy
                m[3] = T[..., 3] + T[..., 0] * uz
                m[4] = T[..., 4] + T[..., 2] * ux + T[..., 1] * uy + T[..., 0] * ux * uy
                m[5] = T[..., 5] + T[..., 3] * ux + T[..., 1] * uz + T[..., 0] * ux * uz
                m[6] = T[..., 6] + T[..., 3] * uy + T[..., 2] * uz + T[..., 0] * uy * uz
                m[7] = T[..., 0] * ux**2 + 2 * T[..., 1] * ux + T[..., 7]
                m[8] = T[..., 0] * uy**2 + 2 * T[..., 2] * uy + T[..., 8]
                m[9] = T[..., 0] * uz**2 + 2 * T[..., 3] * uz + T[..., 9]
                m[10] = T[..., 10] + T[..., 8] * ux + 2 * T[..., 4] * uy + T[..., 1] * uy**2 + T[..., 0] * ux * uy**2 + 2 * T[..., 2] * ux * uy
                m[11] = T[..., 11] + T[..., 9] * ux + 2 * T[..., 5] * uz + T[..., 1] * uz**2 + T[..., 0] * ux * uz**2 + 2 * T[..., 3] * ux * uz
                m[12] = T[..., 12] + 2 * T[..., 4] * ux + T[..., 7] * uy + T[..., 2] * ux**2 + T[..., 0] * ux**2 * uy + 2 * T[..., 1] * ux * uy
                m[13] = T[..., 13] + 2 * T[..., 5] * ux + T[..., 7] * uz + T[..., 3] * ux**2 + T[..., 0] * ux**2 * uz + 2 * T[..., 1] * ux * uz
                m[14] = T[..., 14] + T[..., 9] * uy + 2 * T[..., 6] * uz + T[..., 2] * uz**2 + T[..., 0] * uy * uz**2 + 2 * T[..., 3] * uy * uz
                m[15] = T[..., 15] + 2 * T[..., 6] * uy + T[..., 8] * uz + T[..., 3] * uy**2 + T[..., 0] * uy**2 * uz + 2 * T[..., 2] * uy * uz
                m[16] = (
                    T[..., 16]
                    + T[..., 6] * ux
                    + T[..., 5] * uy
                    + T[..., 4] * uz
                    + T[..., 3] * ux * uy
                    + T[..., 2] * ux * uz
                    + T[..., 1] * uy * uz
                    + T[..., 0] * ux * uy * uz
                )
                m[17] = (
                    T[..., 0] * ux**2 * uy**2
                    + 2 * T[..., 2] * ux**2 * uy
                    + T[..., 8] * ux**2
                    + 2 * T[..., 1] * ux * uy**2
                    + 4 * T[..., 4] * ux * uy
                    + 2 * T[..., 10] * ux
                    + T[..., 7] * uy**2
                    + 2 * T[..., 12] * uy
                    + T[..., 17]
                )
                m[18] = (
                    T[..., 0] * ux**2 * uz**2
                    + 2 * T[..., 3] * ux**2 * uz
                    + T[..., 9] * ux**2
                    + 2 * T[..., 1] * ux * uz**2
                    + 4 * T[..., 5] * ux * uz
                    + 2 * T[..., 11] * ux
                    + T[..., 7] * uz**2
                    + 2 * T[..., 13] * uz
                    + T[..., 18]
                )
                m[19] = (
                    T[..., 0] * uy**2 * uz**2
                    + 2 * T[..., 3] * uy**2 * uz
                    + T[..., 9] * uy**2
                    + 2 * T[..., 2] * uy * uz**2
                    + 4 * T[..., 6] * uy * uz
                    + 2 * T[..., 14] * uy
                    + T[..., 8] * uz**2
                    + 2 * T[..., 15] * uz
                    + T[..., 19]
                )
                m[20] = (
                    T[..., 20]
                    + 2 * T[..., 16] * ux
                    + T[..., 13] * uy
                    + T[..., 12] * uz
                    + T[..., 6] * ux**2
                    + T[..., 3] * ux**2 * uy
                    + T[..., 2] * ux**2 * uz
                    + 2 * T[..., 5] * ux * uy
                    + 2 * T[..., 4] * ux * uz
                    + T[..., 7] * uy * uz
                    + 2 * T[..., 1] * ux * uy * uz
                    + T[..., 0] * ux**2 * uy * uz
                )
                m[21] = (
                    T[..., 21]
                    + T[..., 15] * ux
                    + 2 * T[..., 16] * uy
                    + T[..., 10] * uz
                    + T[..., 5] * uy**2
                    + T[..., 3] * ux * uy**2
                    + T[..., 1] * uy**2 * uz
                    + 2 * T[..., 6] * ux * uy
                    + T[..., 8] * ux * uz
                    + 2 * T[..., 4] * uy * uz
                    + 2 * T[..., 2] * ux * uy * uz
                    + T[..., 0] * ux * uy**2 * uz
                )
                m[22] = (
                    T[..., 22]
                    + T[..., 14] * ux
                    + T[..., 11] * uy
                    + 2 * T[..., 16] * uz
                    + T[..., 4] * uz**2
                    + T[..., 2] * ux * uz**2
                    + T[..., 1] * uy * uz**2
                    + T[..., 9] * ux * uy
                    + 2 * T[..., 6] * ux * uz
                    + 2 * T[..., 5] * uy * uz
                    + 2 * T[..., 3] * ux * uy * uz
                    + T[..., 0] * ux * uy * uz**2
                )
                m[23] = (
                    T[..., 23]
                    + T[..., 19] * ux
                    + 2 * T[..., 22] * uy
                    + 2 * T[..., 21] * uz
                    + T[..., 11] * uy**2
                    + T[..., 10] * uz**2
                    + T[..., 9] * ux * uy**2
                    + T[..., 8] * ux * uz**2
                    + 2 * T[..., 4] * uy * uz**2
                    + 2 * T[..., 5] * uy**2 * uz
                    + T[..., 1] * uy**2 * uz**2
                    + 2 * T[..., 14] * ux * uy
                    + 2 * T[..., 15] * ux * uz
                    + 4 * T[..., 16] * uy * uz
                    + 4 * T[..., 6] * ux * uy * uz
                    + 2 * T[..., 2] * ux * uy * uz**2
                    + 2 * T[..., 3] * ux * uy**2 * uz
                    + T[..., 0] * ux * uy**2 * uz**2
                )
                m[24] = (
                    T[..., 24]
                    + 2 * T[..., 22] * ux
                    + T[..., 18] * uy
                    + 2 * T[..., 20] * uz
                    + T[..., 14] * ux**2
                    + T[..., 12] * uz**2
                    + T[..., 9] * ux**2 * uy
                    + 2 * T[..., 4] * ux * uz**2
                    + 2 * T[..., 6] * ux**2 * uz
                    + T[..., 7] * uy * uz**2
                    + T[..., 2] * ux**2 * uz**2
                    + 2 * T[..., 11] * ux * uy
                    + 4 * T[..., 16] * ux * uz
                    + 2 * T[..., 13] * uy * uz
                    + 4 * T[..., 5] * ux * uy * uz
                    + 2 * T[..., 1] * ux * uy * uz**2
                    + 2 * T[..., 3] * ux**2 * uy * uz
                    + T[..., 0] * ux**2 * uy * uz**2
                )
                m[25] = (
                    T[..., 25]
                    + 2 * T[..., 21] * ux
                    + 2 * T[..., 20] * uy
                    + T[..., 17] * uz
                    + T[..., 15] * ux**2
                    + T[..., 13] * uy**2
                    + 2 * T[..., 5] * ux * uy**2
                    + 2 * T[..., 6] * ux**2 * uy
                    + T[..., 8] * ux**2 * uz
                    + T[..., 7] * uy**2 * uz
                    + T[..., 3] * ux**2 * uy**2
                    + 4 * T[..., 16] * ux * uy
                    + 2 * T[..., 10] * ux * uz
                    + 2 * T[..., 12] * uy * uz
                    + 4 * T[..., 4] * ux * uy * uz
                    + 2 * T[..., 1] * ux * uy**2 * uz
                    + 2 * T[..., 2] * ux**2 * uy * uz
                    + T[..., 0] * ux**2 * uy**2 * uz
                )
                m[26] = (
                    T[..., 0] * ux**2 * uy**2 * uz**2
                    + 2 * T[..., 3] * ux**2 * uy**2 * uz
                    + T[..., 9] * ux**2 * uy**2
                    + 2 * T[..., 2] * ux**2 * uy * uz**2
                    + 4 * T[..., 6] * ux**2 * uy * uz
                    + 2 * T[..., 14] * ux**2 * uy
                    + T[..., 8] * ux**2 * uz**2
                    + 2 * T[..., 15] * ux**2 * uz
                    + T[..., 19] * ux**2
                    + 2 * T[..., 1] * ux * uy**2 * uz**2
                    + 4 * T[..., 5] * ux * uy**2 * uz
                    + 2 * T[..., 11] * ux * uy**2
                    + 4 * T[..., 4] * ux * uy * uz**2
                    + 8 * T[..., 16] * ux * uy * uz
                    + 4 * T[..., 22] * ux * uy
                    + 2 * T[..., 10] * ux * uz**2
                    + 4 * T[..., 21] * ux * uz
                    + 2 * T[..., 23] * ux
                    + T[..., 7] * uy**2 * uz**2
                    + 2 * T[..., 13] * uy**2 * uz
                    + T[..., 18] * uy**2
                    + 2 * T[..., 12] * uy * uz**2
                    + 4 * T[..., 20] * uy * uz
                    + 2 * T[..., 24] * uy
                    + T[..., 17] * uz**2
                    + 2 * T[..., 25] * uz
                    + T[..., 26]
                )
                m = stack_components(m, T)
                return m

            return shift_inverse(T, u)

    @partial(jit, static_argnums=(0,))
    def compute_eq_central_moments(self, rho):
        """
        Calculate the central moments of the equilibrium distribution.

        Parameters
        ----------
        rho (jax.numpy.ndarray): Density field.

        Returns
        -------
        T_eq : jax.numpy.ndarray
            Central moments of the equilibrium distribution.
        """

        if isinstance(self.lattice, LatticeD2Q9):
            T_eq = {}
            T_eq[0] = rho[..., 0]
            T_eq[3] = 2 * rho[..., 0] * self.lattice.cs2
            T_eq[8] = rho[..., 0] * self.lattice.cs**4
            T_eq = stack_components(T_eq, jnp.zeros((self.nx, self.ny, self.lattice.q), dtype=self.precision_policy.compute_dtype))

            return T_eq

        elif isinstance(self.lattice, LatticeD3Q19):
            T_eq = {}
            T_eq[0] = rho[..., 0]
            T_eq[7] = rho[..., 0] * self.lattice.cs2
            T_eq[8] = rho[..., 0] * self.lattice.cs2
            T_eq[9] = rho[..., 0] * self.lattice.cs2
            T_eq[16] = rho[..., 0] * self.lattice.cs**4
            T_eq[17] = rho[..., 0] * self.lattice.cs**4
            T_eq[18] = rho[..., 0] * self.lattice.cs**4
            T_eq = stack_components(T_eq, jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precision_policy.compute_dtype))

            return T_eq

        elif isinstance(self.lattice, LatticeD3Q27):
            T_eq = {}
            T_eq[0] = rho[..., 0]
            T_eq[7] = rho[..., 0] * self.lattice.cs2
            T_eq[8] = rho[..., 0] * self.lattice.cs2
            T_eq[9] = rho[..., 0] * self.lattice.cs2
            T_eq[17] = rho[..., 0] * self.lattice.cs**4
            T_eq[18] = rho[..., 0] * self.lattice.cs**4
            T_eq[19] = rho[..., 0] * self.lattice.cs**4
            T_eq[26] = rho[..., 0] * self.lattice.cs**6
            T_eq = stack_components(T_eq, jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precision_policy.compute_dtype))

            return T_eq

    @partial(jit, static_argnums=(0,))
    def compute_force_central_moments(self, F):
        """
        Calculate the central moments of the force distribution. Includes modification to accurately replicate mechanical stability conditions.

        Parameters
        ----------
        F (pytree of jax.numpy.ndarray): Force field.

        Returns
        -------
        C : pytree of jax.numpy.ndarray
            Central moments of the force distribution.
        """

        if isinstance(self.lattice, LatticeD2Q9):
            C = {}
            Fx = F[..., 0]
            Fy = F[..., 1]
            C[1] = Fx
            C[2] = Fy
            C[6] = Fy * self.lattice.cs2
            C[7] = Fx * self.lattice.cs2
            C = stack_components(C, jnp.zeros((self.nx, self.ny, self.lattice.q), dtype=self.precision_policy.compute_dtype))

            return C
        elif isinstance(self.lattice, LatticeD3Q19):
            C = {}
            Fx = F[..., 0]
            Fy = F[..., 1]
            Fz = F[..., 2]
            C[1] = Fx
            C[2] = Fy
            C[3] = Fz
            C[10] = Fx * self.lattice.cs2
            C[11] = Fx * self.lattice.cs2
            C[12] = Fy * self.lattice.cs2
            C[13] = Fz * self.lattice.cs2
            C[14] = Fy * self.lattice.cs2
            C[15] = Fz * self.lattice.cs2
            C = stack_components(C, jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precision_policy.compute_dtype))

            return C
        elif isinstance(self.lattice, LatticeD3Q27):
            C = {}
            Fx = F[..., 0]
            Fy = F[..., 1]
            Fz = F[..., 2]
            C[1] = Fx
            C[2] = Fy
            C[3] = Fz
            C[10] = Fx * self.lattice.cs2
            C[11] = Fx * self.lattice.cs2
            C[12] = Fy * self.lattice.cs2
            C[13] = Fz * self.lattice.cs2
            C[14] = Fy * self.lattice.cs2
            C[15] = Fz * self.lattice.cs2
            C[23] = Fx * self.lattice.cs**4
            C[24] = Fy * self.lattice.cs**4
            C[25] = Fz * self.lattice.cs**4
            C = stack_components(C, jnp.zeros((self.nx, self.ny, self.nz, self.lattice.q), dtype=self.precision_policy.compute_dtype))

            return C

    @partial(jit, static_argnums=(0,), inline=True)
    def apply_force(self, Tdash, rho, u):
        """
        Modified version of the apply_force defined in LBMBase to account for modified force.

        Parameters
        ----------
            Tdash (jax.numpy.ndarray): Central moments post-collision distribution functions.

            rho (jax.numpy.ndarray): Density field.

            u (jax.numpy.ndarray): Velocity field.

        Returns
        -------
            f_postcollision (jax.numpy.ndarray): Post-collision distribution functions with the force applied.
        """
        F = self.get_force()
        if F is None:
            F = jnp.zeros_like(u)
        C = self.compute_force_central_moments(F)
        Tf = jnp.dot(C, jnp.eye(self.lattice.q) - 0.5 * self.S)
        return Tdash + Tf

    # Direction-major populations in run() by default in 3D (+17%; see LBMBase._direction_major_step); not measured in 2D
    _direction_major_dims = (3,)

    _CASCADE_TRANSFORMS = (
        "compute_central_moment",
        "compute_central_moment_inverse",
        "compute_eq_central_moments",
        "compute_force_central_moments",
        "apply_force",
    )

    def _collision_applies_bounceback(self):
        """The per-direction collision applies the full-way bounce-back itself (see LBMBase.apply_fused_bounceback)."""
        return type(self).collision is CLBMSim.collision and self._uses_default_cascade_transforms()

    def _uses_default_cascade_transforms(self):
        """Whether all moment transforms used by the collision are this class's own (then the component-mode collision applies)."""
        return all(getattr(type(self), name) is getattr(CLBMSim, name) for name in self._CASCADE_TRANSFORMS)

    def _collision_components(self, fin, rho, u):
        """
        The cascaded collision evaluated per lattice direction (component mode), for the default moment transforms.

        Same algorithm as the array path of collision(), with the (..., q) intermediates kept as q separate arrays (Components),
        the same transform methods traced in component mode, and the matrix products as sparse sums with constant coefficients.
        XLA then compiles the whole collision into one kernel with one thread per node. The products are exact float32/float64
        sums, so results can differ from the GEMM path at roundoff (or TF32) level. See MultiphaseCascade._collision_components.
        """
        dtype = self.precision_policy.compute_dtype
        q = self.lattice.q
        eye = np.eye(q)
        S = np.asarray(self.S)
        transform = lambda name: getattr(CLBMSim, name).__wrapped__
        F = self.get_force()
        if F is None:
            F = jnp.zeros((*fin.shape[:-1], self.dim), dtype=dtype)
        with component_mode():
            m = matvec_components(Components([fin[..., i] for i in range(q)], dtype), self.sparse_collision_terms(np.asarray(self.M)), dtype)
            Tdash = transform("compute_central_moment")(self, m, u)
            Tdash_eq = transform("compute_eq_central_moments")(self, rho)
            C = transform("compute_force_central_moments")(self, F)
            relaxed = matvec_components(Tdash, self.sparse_collision_terms(eye - S), dtype)
            equilibrium = matvec_components(Tdash_eq, self.sparse_collision_terms(S), dtype)
            forcing = matvec_components(C, self.sparse_collision_terms(eye - 0.5 * S), dtype)
            Tout = Components([a + b + c for a, b, c in zip(relaxed, equilibrium, forcing, strict=True)], dtype)
            m_out = transform("compute_central_moment_inverse")(self, Tout, u)
            fout = matvec_components(m_out, self.sparse_collision_terms(np.asarray(self.M_inv)), dtype)
        outputs = self.apply_fused_bounceback(
            [jnp.broadcast_to(x, fin.shape[:-1]) for x in fout], [fin[..., i] for i in range(q)], self.bounceback_mask
        )
        return self.stack_populations([self.precision_policy.cast_to_output(x) for x in outputs])

    @partial(jit, static_argnums=(0,))
    def collision(self, fin):
        """
        Cascaded LBM collision step for lattice.
        """
        fin = self.precision_policy.cast_to_compute(fin)
        if self._uses_default_cascade_transforms():
            cls = type(self)
            if cls.update_macroscopic is LBMBase.update_macroscopic and cls.macroscopic_velocity is CLBMSim.macroscopic_velocity:
                # Density and velocity per direction too (as Components, read by the transforms as rho[..., 0], u[..., a])
                dtype = self.precision_policy.compute_dtype
                _, _, _, rho, velocity = _per_direction_moments(self.lattice, fin)
                if self.force is not None:
                    velocity = [v + 0.5 * self.force[..., axis] / rho for axis, v in enumerate(velocity)]
                return self._collision_components(fin, Components([rho], dtype), Components(velocity, dtype))
            rho, _ = self.update_macroscopic(fin)
            return self._collision_components(fin, rho, self.macroscopic_velocity(fin, rho))
        rho, _ = self.update_macroscopic(fin)
        u = self.macroscopic_velocity(fin, rho)
        T = jnp.dot(fin, self.M)
        Tdash = self.compute_central_moment(T, u)
        Tdash_eq = self.compute_eq_central_moments(rho)
        Tout = jnp.dot(Tdash, jnp.eye(self.lattice.q) - self.S) + jnp.dot(Tdash_eq, self.S)
        Tout = self.apply_force(Tout, rho, u)
        Tout = self.compute_central_moment_inverse(Tout, u)
        fout = jnp.dot(Tout, self.M_inv)
        return self.precision_policy.cast_to_output(fout)

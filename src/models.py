import jax.numpy as jnp
from jax import jit
from functools import partial
from src.base import LBMBase
"""
Collision operators are defined in this file for different models.
"""

class BGKSim(LBMBase):
    """
    BGK simulation class.

    This class implements the Bhatnagar-Gross-Krook (BGK) approximation for the collision step in the Lattice Boltzmann Method.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f):
        """
        BGK collision step for lattice.

        The collision step is where the main physics of the LBM is applied. In the BGK approximation, 
        the distribution function is relaxed towards the equilibrium distribution function.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        rho, u = self.update_macroscopic(f)
        feq = self.equilibrium(rho, u, cast_output=False)
        fneq = f - feq
        fout = f - self.omega * fneq
        if self.force is not None:
            fout = self.apply_force(fout, feq, rho, u)
        return self.precisionPolicy.cast_to_output(fout)

class KBCSim(LBMBase):
    """
    KBC simulation class.

    This class implements the Karlin-Bösch-Chikatamarla (KBC) model for the collision step in the Lattice Boltzmann Method.
    """
    def __init__(self, **kwargs):
        if kwargs.get('lattice').name != 'D3Q27' and kwargs.get('nz') > 0:
            raise ValueError("KBC collision operator in 3D must only be used with D3Q27 lattice.")
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f):
        """
        KBC collision step for lattice.
        """
        f = self.precisionPolicy.cast_to_compute(f)
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
        gamma = invBeta - (2.0 - invBeta) * self.entropic_scalar_product(deltaS, deltaH, feq) / (tiny + self.entropic_scalar_product(deltaH, deltaH, feq))

        fout = f - beta * (2.0 * deltaS + gamma[..., None] * deltaH)

        # add external force
        if self.force is not None:
            fout = self.apply_force(fout, feq, rho, u)
        return self.precisionPolicy.cast_to_output(fout)

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
        fneq : jax.numpy.array
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.array
            Shear components of fneq.
        """
        Pi = self.momentum_flux(fneq)
        N = Pi[..., 0] - Pi[..., 2]
        s = jnp.zeros_like(fneq)
        s = s.at[..., 6].set(N)
        s = s.at[..., 3].set(N)
        s = s.at[..., 2].set(-N)
        s = s.at[..., 1].set(-N)
        s = s.at[..., 8].set(Pi[..., 1])
        s = s.at[..., 4].set(-Pi[..., 1])
        s = s.at[..., 5].set(-Pi[..., 1])
        s = s.at[..., 7].set(Pi[..., 1])

        return s

    @partial(jit, static_argnums=(0,), inline=True)
    def fdecompose_shear_d3q27(self, fneq):
        """
        Decompose fneq into shear components for D3Q27 lattice.

        Parameters
        ----------
        fneq : jax.numpy.ndarray
            Non-equilibrium distribution function.

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
        Pi = self.momentum_flux(fneq)
        Nxz = Pi[..., 0] - Pi[..., 5]
        Nyz = Pi[..., 3] - Pi[..., 5]

        # For c = (i, 0, 0), c = (0, j, 0) and c = (0, 0, k)
        s = jnp.zeros_like(fneq)
        s = s.at[..., 9].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[..., 18].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[..., 3].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[..., 6].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[..., 1].set((-Nxz - Nyz) / 6.0)
        s = s.at[..., 2].set((-Nxz - Nyz) / 6.0)

        # For c = (i, j, 0)
        s = s.at[..., 12].set(Pi[..., 1] / 4.0)
        s = s.at[..., 24].set(Pi[..., 1] / 4.0)
        s = s.at[..., 21].set(-Pi[..., 1] / 4.0)
        s = s.at[..., 15].set(-Pi[..., 1] / 4.0)

        # For c = (i, 0, k)
        s = s.at[..., 10].set(Pi[..., 2] / 4.0)
        s = s.at[..., 20].set(Pi[..., 2] / 4.0)
        s = s.at[..., 19].set(-Pi[..., 2] / 4.0)
        s = s.at[..., 11].set(-Pi[..., 2] / 4.0)

        # For c = (0, j, k)
        s = s.at[..., 8].set(Pi[..., 4] / 4.0)
        s = s.at[..., 4].set(Pi[..., 4] / 4.0)
        s = s.at[..., 7].set(-Pi[..., 4] / 4.0)
        s = s.at[..., 5].set(-Pi[..., 4] / 4.0)

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

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def collision(self, f):
        """
        BGK collision step for lattice.
        """
        f = self.precisionPolicy.cast_to_compute(f)
        rho =jnp.sum(f, axis=-1, keepdims=True)
        feq = self.equilibrium(rho, self.vel, cast_output=False)
        fneq = f - feq
        fout = f - self.omega * fneq
        return self.precisionPolicy.cast_to_output(fout)
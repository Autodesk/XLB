# Base Velocity Set class

import math
import numpy as np
from functools import partial
import jax.numpy as jnp
from jax import jit, vmap
import numba
from numba import cuda, float32, int32


class VelocitySet(object):
    """
    Base class for the velocity set of the Lattice Boltzmann Method (LBM), e.g. D2Q9, D3Q27, etc.

    Parameters
    ----------
    d: int
        The dimension of the lattice.
    q: int
        The number of velocities of the lattice.
    c: numpy.ndarray
        The velocity vectors of the lattice. Shape: (q, d)
    w: numpy.ndarray
        The weights of the lattice. Shape: (q,)
    """

    def __init__(self, d, q, c, w):
        # Store the dimension and the number of velocities
        self.d = d
        self.q = q

        # Constants
        self.cs = math.sqrt(3) / 3.0
        self.cs2 = 1.0 / 3.0
        self.inv_cs2 = 3.0

        # Construct the properties of the lattice
        self.c = c
        self.w = w
        self.cc = self._construct_lattice_moment()
        self.opp_indices = self._construct_opposite_indices()
        self.main_indices = self._construct_main_indices()
        self.right_indices = self._construct_right_indices()
        self.left_indices = self._construct_left_indices()

    @partial(jit, static_argnums=(0,))
    def momentum_flux_jax(self, fneq):
        """
        This function computes the momentum flux, which is the product of the non-equilibrium
        distribution functions (fneq) and the lattice moments (cc).

        The momentum flux is used in the computation of the stress tensor in the Lattice Boltzmann
        Method (LBM).

        Parameters
        ----------
        fneq: jax.numpy.ndarray
            The non-equilibrium distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The computed momentum flux.
        """

        return jnp.dot(fneq, self.cc)

    def momentum_flux_numba(self):
        """
        This function computes the momentum flux, which is the product of the non-equilibrium
        """
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def decompose_shear_jax(self, fneq):
        """
        Decompose fneq into shear components for D3Q27 lattice.

        TODO: add generali

        Parameters
        ----------
        fneq : jax.numpy.ndarray
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.ndarray
            Shear components of fneq.
        """
        raise NotImplementedError

    def decompose_shear_numba(self):
        """
        Decompose fneq into shear components for D3Q27 lattice.
        """
        raise NotImplementedError

    def _construct_lattice_moment(self):
        """
        This function constructs the moments of the lattice.

        The moments are the products of the velocity vectors, which are used in the computation of
        the equilibrium distribution functions and the collision operator in the Lattice Boltzmann
        Method (LBM).

        Returns
        -------
        cc: numpy.ndarray
            The moments of the lattice.
        """
        c = self.c.T
        # Counter for the loop
        cntr = 0

        # nt: number of independent elements of a symmetric tensor
        nt = self.d * (self.d + 1) // 2

        cc = np.zeros((self.q, nt))
        for a in range(0, self.d):
            for b in range(a, self.d):
                cc[:, cntr] = c[:, a] * c[:, b]
                cntr += 1

        return cc

    def _construct_opposite_indices(self):
        """
        This function constructs the indices of the opposite velocities for each velocity.

        The opposite velocity of a velocity is the velocity that has the same magnitude but the
        opposite direction.

        Returns
        -------
        opposite: numpy.ndarray
            The indices of the opposite velocities.
        """
        c = self.c.T
        opposite = np.array([c.tolist().index((-c[i]).tolist()) for i in range(self.q)])
        return opposite

    def _construct_main_indices(self):
        """
        This function constructs the indices of the main velocities.

        The main velocities are the velocities that have a magnitude of 1 in lattice units.

        Returns
        -------
        numpy.ndarray
            The indices of the main velocities.
        """
        c = self.c.T
        if self.d == 2:
            return np.nonzero((np.abs(c[:, 0]) + np.abs(c[:, 1]) == 1))[0]

        elif self.d == 3:
            return np.nonzero(
                (np.abs(c[:, 0]) + np.abs(c[:, 1]) + np.abs(c[:, 2]) == 1)
            )[0]

    def _construct_right_indices(self):
        """
        This function constructs the indices of the velocities that point in the positive
        x-direction.

        Returns
        -------
        numpy.ndarray
            The indices of the right velocities.
        """
        c = self.c.T
        return np.nonzero(c[:, 0] == 1)[0]

    def _construct_left_indices(self):
        """
        This function constructs the indices of the velocities that point in the negative
        x-direction.

        Returns
        -------
        numpy.ndarray
            The indices of the left velocities.
        """
        c = self.c.T
        return np.nonzero(c[:, 0] == -1)[0]

    def __str__(self):
        """
        This function returns the name of the lattice in the format of DxQy.
        """
        return self.__repr__()

    def __repr__(self):
        """
        This function returns the name of the lattice in the format of DxQy.
        """
        return "D{}Q{}".format(self.d, self.q)

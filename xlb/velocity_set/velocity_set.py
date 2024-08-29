# Base Velocity Set class

import math
import numpy as np

import warp as wp


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
        self.get_opp_index = lambda i: self.opp_indices[i]
        self.main_indices = self._construct_main_indices()
        self.right_indices = self._construct_right_indices()
        self.left_indices = self._construct_left_indices()
        self.qi = self._construct_qi()

        # Make warp constants for these vectors
        # TODO: Following warp updates these may not be necessary
        self.wp_c = wp.constant(wp.mat((self.d, self.q), dtype=wp.int32)(self.c))
        self.wp_w = wp.constant(wp.vec(self.q, dtype=wp.float32)(self.w))  # TODO: Make type optional somehow
        self.wp_opp_indices = wp.constant(wp.vec(self.q, dtype=wp.int32)(self.opp_indices))
        self.wp_cc = wp.constant(wp.mat((self.q, self.d * (self.d + 1) // 2), dtype=wp.float32)(self.cc))
        self.wp_c32 = wp.constant(wp.mat((self.d, self.q), dtype=wp.float32)(self.c))
        self.wp_qi = wp.constant(wp.mat((self.q, self.d * (self.d + 1) // 2), dtype=wp.float32)(self.qi))

    def warp_lattice_vec(self, dtype):
        return wp.vec(len(self.c), dtype=dtype)

    def warp_u_vec(self, dtype):
        return wp.vec(self.d, dtype=dtype)

    def warp_stream_mat(self, dtype):
        return wp.mat((self.q, self.d), dtype=dtype)

    def _construct_qi(self):
        # Qi = cc - cs^2*I
        dim = self.d
        Qi = self.cc.copy()
        if dim == 3:
            diagonal = (0, 3, 5)
            offdiagonal = (1, 2, 4)
        elif dim == 2:
            diagonal = (0, 2)
            offdiagonal = (1,)
        else:
            raise ValueError(f"dim = {dim} not supported")

        # multiply off-diagonal elements by 2 because the Q tensor is symmetric
        Qi[:, diagonal] += -1.0 / 3.0
        Qi[:, offdiagonal] *= 2.0
        return Qi

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
            return np.nonzero((np.abs(c[:, 0]) + np.abs(c[:, 1]) + np.abs(c[:, 2]) == 1))[0]

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

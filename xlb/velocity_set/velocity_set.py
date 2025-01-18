# Base Velocity Set class

import math
import numpy as np
import warp as wp
import jax.numpy as jnp
import jax

from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy


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

    def __init__(self, d, q, c, w, precision_policy, compute_backend):
        # Store the dimension and the number of velocities
        self.d = d
        self.q = q
        self.precision_policy = precision_policy
        self.compute_backend = compute_backend

        # Updating JAX config in case fp64 is requested
        if compute_backend == ComputeBackend.JAX and (precision_policy == PrecisionPolicy.FP64FP64 or precision_policy == PrecisionPolicy.FP64FP32):
            jax.config.update("jax_enable_x64", True)

        # Create all properties in NumPy first
        self._init_numpy_properties(c, w)

        # Convert properties to backend-specific format
        if self.compute_backend == ComputeBackend.WARP:
            self._init_warp_properties()
        elif self.compute_backend == ComputeBackend.JAX:
            self._init_jax_properties()
        else:
            raise ValueError(f"Unsupported compute backend: {self.compute_backend}")

        # Set up backend-specific constants
        self._init_backend_constants()

    def _init_numpy_properties(self, c, w):
        """
        Initialize all properties in NumPy first.
        """
        self._c = np.array(c)
        self._w = np.array(w)
        self._opp_indices = self._construct_opposite_indices()
        self._cc = self._construct_lattice_moment()
        self._c_float = self._c.astype(np.float64)
        self._qi = self._construct_qi()

        # Constants in NumPy
        self.cs = np.float64(math.sqrt(3) / 3.0)
        self.cs2 = np.float64(1.0 / 3.0)
        self.inv_cs2 = np.float64(3.0)

        # Indices
        self.main_indices = self._construct_main_indices()
        self.right_indices = self._construct_right_indices()
        self.left_indices = self._construct_left_indices()

    def _init_warp_properties(self):
        """
        Convert NumPy properties to Warp-specific properties.
        """
        dtype = self.precision_policy.compute_precision.wp_dtype
        self.c = wp.constant(wp.mat((self.d, self.q), dtype=wp.int32)(self._c))
        self.w = wp.constant(wp.vec(self.q, dtype=dtype)(self._w))
        self.opp_indices = wp.constant(wp.vec(self.q, dtype=wp.int32)(self._opp_indices))
        self.cc = wp.constant(wp.mat((self.q, self.d * (self.d + 1) // 2), dtype=dtype)(self._cc))
        self.c_float = wp.constant(wp.mat((self.d, self.q), dtype=dtype)(self._c_float))
        self.qi = wp.constant(wp.mat((self.q, self.d * (self.d + 1) // 2), dtype=dtype)(self._qi))

    def _init_jax_properties(self):
        """
        Convert NumPy properties to JAX-specific properties.
        """
        dtype = self.precision_policy.compute_precision.jax_dtype
        self.c = jnp.array(self._c, dtype=jnp.int32)
        self.w = jnp.array(self._w, dtype=dtype)
        self.opp_indices = jnp.array(self._opp_indices, dtype=jnp.int32)
        self.cc = jnp.array(self._cc, dtype=dtype)
        self.c_float = jnp.array(self._c_float, dtype=dtype)
        self.qi = jnp.array(self._qi, dtype=dtype)

    def _init_backend_constants(self):
        """
        Initialize the constants for the backend.
        """
        if self.compute_backend == ComputeBackend.WARP:
            dtype = self.precision_policy.compute_precision.wp_dtype
            self.cs = wp.constant(dtype(self.cs))
            self.cs2 = wp.constant(dtype(self.cs2))
            self.inv_cs2 = wp.constant(dtype(self.inv_cs2))
        elif self.compute_backend == ComputeBackend.JAX:
            dtype = self.precision_policy.compute_precision.jax_dtype
            self.cs = jnp.array(self.cs, dtype=dtype)
            self.cs2 = jnp.array(self.cs2, dtype=dtype)
            self.inv_cs2 = jnp.array(self.inv_cs2, dtype=dtype)

    def warp_lattice_vec(self, dtype):
        return wp.vec(len(self.c), dtype=dtype)

    def warp_u_vec(self, dtype):
        return wp.vec(self.d, dtype=dtype)

    def warp_stream_mat(self, dtype):
        return wp.mat((self.q, self.d), dtype=dtype)

    def _construct_qi(self):
        # Qi = cc - cs^2*I
        dim = self.d
        Qi = self._cc.copy()
        if dim == 3:
            diagonal, offdiagonal = (0, 3, 5), (1, 2, 4)
        elif dim == 2:
            diagonal, offdiagonal = (0, 2), (1,)
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
        c = self._c.T
        # Counter for the loop
        cntr = 0
        c = self._c.T
        # nt: number of independent elements of a symmetric tensor
        nt = self.d * (self.d + 1) // 2
        cc = np.zeros((self.q, nt))
        cntr = 0
        for a in range(self.d):
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
        c = self._c.T
        return np.array([c.tolist().index((-c[i]).tolist()) for i in range(self.q)])

    def _construct_main_indices(self):
        """
        This function constructs the indices of the main velocities.

        The main velocities are the velocities that have a magnitude of 1 in lattice units.

        Returns
        -------
        numpy.ndarray
            The indices of the main velocities.
        """
        c = self._c.T
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
        return np.nonzero(self._c.T[:, 0] == 1)[0]

    def _construct_left_indices(self):
        """
        This function constructs the indices of the velocities that point in the negative
        x-direction.

        Returns
        -------
        numpy.ndarray
            The indices of the left velocities.
        """
        return np.nonzero(self._c.T[:, 0] == -1)[0]

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

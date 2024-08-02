# Description: Lattice class for 3D D3Q27 lattice.

import itertools
import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet


class D3Q27(VelocitySet):
    """
    Velocity Set for 3D D3Q27 lattice.

    D3Q27 stands for three-dimensional twenty-seven-velocity model. It is a common model used in the
    Lattice Boltzmann Method for simulating fluid flows in three dimensions.
    """

    def __init__(self):
        # Construct the velocity vectors and weights
        c = np.array(list(itertools.product([0, -1, 1], repeat=3))).T
        w = np.zeros(27)
        for i in range(27):
            if np.sum(np.abs(c[:, i])) == 0:
                w[i] = 8.0 / 27.0
            elif np.sum(np.abs(c[:, i])) == 1:
                w[i] = 2.0 / 27.0
            elif np.sum(np.abs(c[:, i])) == 2:
                w[i] = 1.0 / 54.0
            elif np.sum(np.abs(c[:, i])) == 3:
                w[i] = 1.0 / 216.0

        # Initialize the Lattice
        super().__init__(3, 27, c, w)

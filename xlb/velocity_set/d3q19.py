# Description: Lattice class for 3D D3Q19 lattice.

import itertools
import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet


class D3Q19(VelocitySet):
    """
    Velocity Set for 3D D3Q19 lattice.

    D3Q19 stands for three-dimensional nineteen-velocity model. It is a common model used in the
    Lattice Boltzmann Method for simulating fluid flows in three dimensions.
    """

    def __init__(self):
        # Construct the velocity vectors and weights
        c = np.array(
            [
                ci
                for ci in itertools.product([-1, 0, 1], repeat=3)
                if np.sum(np.abs(ci)) <= 2
            ]
        ).T
        w = np.zeros(19)
        for i in range(19):
            if np.sum(np.abs(c[:, i])) == 0:
                w[i] = 1.0 / 3.0
            elif np.sum(np.abs(c[:, i])) == 1:
                w[i] = 1.0 / 18.0
            elif np.sum(np.abs(c[:, i])) == 2:
                w[i] = 1.0 / 36.0

        # Initialize the lattice
        super().__init__(3, 19, c, w)

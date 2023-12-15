# Description: Lattice class for 2D D2Q9 lattice.

import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet


class D2Q9(VelocitySet):
    """
    Velocity Set for 2D D2Q9 lattice.

    D2Q9 stands for two-dimensional nine-velocity model. It is a common model used in the
    Lat tice Boltzmann Method for simulating fluid flows in two dimensions.
    """

    def __init__(self):
        # Construct the velocity vectors and weights
        cx = [0, 0, 0, 1, -1, 1, -1, 1, -1]
        cy = [0, 1, -1, 0, 1, -1, 0, 1, -1]
        c = np.array(tuple(zip(cx, cy))).T
        w = np.array(
            [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 9, 1 / 36, 1 / 36]
        )

        # Call the parent constructor
        super().__init__(2, 9, c, w)

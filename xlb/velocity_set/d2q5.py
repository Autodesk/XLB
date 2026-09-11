# Description: Lattice class for 2D D2Q5 lattice.

import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet


class D2Q5(VelocitySet):
    """
    Velocity Set for 2D D2Q5 lattice.

    D2Q5 stands for two-dimensional five-velocity model. It is the minimal stencil that
    recovers the advection-diffusion equation and is therefore the cheapest choice for a
    passive scalar such as temperature. It cannot be used for Navier-Stokes because its
    fourth-order moments are not isotropic.

    The weights are chosen such that the lattice speed of sound is cs^2 = 1/3, matching
    the value assumed by :class:`VelocitySet` and used by D2Q9 / D3Q19 / D3Q27.
    """

    def __init__(self, precision_policy, compute_backend):
        # Construct the velocity vectors and weights
        cx = [0, 0, 0, 1, -1]
        cy = [0, 1, -1, 0, 0]
        c = np.array(tuple(zip(cx, cy))).T
        w = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])

        # Call the parent constructor
        super().__init__(2, 5, c, w, precision_policy=precision_policy, compute_backend=compute_backend)

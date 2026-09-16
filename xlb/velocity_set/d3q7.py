# Description: Lattice class for 3D D3Q7 lattice.

import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet


class D3Q7(VelocitySet):
    """
    Velocity Set for 3D D3Q7 lattice.

    D3Q7 stands for three-dimensional seven-velocity model. It is the minimal stencil that
    recovers the advection-diffusion equation in 3D and is therefore the cheapest choice for
    a passive scalar such as temperature. It is the three-dimensional counterpart of
    :class:`D2Q5`, and like D2Q5 it cannot be used for Navier-Stokes because its
    fourth-order moments are not isotropic.

    Note on the speed of sound. Unlike every other stencil in XLB, D3Q7 does **not** have
    cs^2 = 1/3. With w_0 = 1/4 for the rest population and w_i = 1/8 for the six
    axis-aligned directions the moments are

        sum_i w_i           = 1
        sum_i w_i c_ia c_ib = cs^2 delta_ab   with   cs^2 = 1/4

    and cs^2 = 1/3 is simply not reachable on this stencil: six axis directions of equal
    weight w give sum_i w_i c_ia c_ib = 2w, so cs^2 = 1/3 would force w = 1/6 and hence
    w_0 = 0, degenerating the rest population.

    :class:`VelocitySet` derives cs^2 from the stencil, so this is handled automatically.
    When choosing a relaxation rate, pass the matching value explicitly::

        omega = omega_from_diffusivity(alpha, cs2=velocity_set.cs2)
    """

    def __init__(self, precision_policy, compute_backend):
        # Construct the velocity vectors and weights
        cx = [0, 0, 0, 0, 0, 1, -1]
        cy = [0, 1, -1, 0, 0, 0, 0]
        cz = [0, 0, 0, 1, -1, 0, 0]
        c = np.array(tuple(zip(cx, cy, cz))).T
        w = np.array([1 / 4, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8])

        # Call the parent constructor
        super().__init__(3, 7, c, w, precision_policy=precision_policy, compute_backend=compute_backend)

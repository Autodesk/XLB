# Purpose: Precision policy for lattice Boltzmann method with computation and
#          storage precision both set to float32.

import jax.numpy as jnp

from xlb.precision_policy.precision_policy import PrecisionPolicy


class Fp32Fp32(PrecisionPolicy):
    """
    Precision policy for lattice Boltzmann method with computation and storage
    precision both set to float32.

    Parameters
    ----------
    None
    """

    def __init__(self):
        super().__init__(jnp.float32, jnp.float32)

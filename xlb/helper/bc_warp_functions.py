from xlb import DefaultConfig
from xlb.compute_backend import ComputeBackend
from xlb.operator.macroscopic.second_moment import SecondMoment as MomentumFlux
import warp as wp
from typing import Any

# Set the compute and Store dtypes
if DefaultConfig.default_backend == ComputeBackend.JAX:
    compute_dtype = DefaultConfig.default_precision_policy.compute_precision.jax_dtype
    store_dtype = DefaultConfig.default_precision_policy.store_precision.jax_dtype
elif DefaultConfig.default_backend == ComputeBackend.WARP:
    compute_dtype = DefaultConfig.default_precision_policy.compute_precision.wp_dtype
    compute_dtype = DefaultConfig.default_precision_policy.store_precision.wp_dtype

# Set local constants
_d = DefaultConfig.velocity_set.d
_q = DefaultConfig.velocity_set.q
_u_vec = wp.vec(_d, dtype=compute_dtype)
_opp_indices = DefaultConfig.velocity_set.opp_indices
_w = DefaultConfig.velocity_set.w
_c = DefaultConfig.velocity_set.c
_c_float = DefaultConfig.velocity_set.c_float
_qi = DefaultConfig.velocity_set.qi


# Define the operator needed for computing the momentum flux
momentum_flux = MomentumFlux()


@wp.func
def get_bc_fsum(
    fpop: Any,
    missing_mask: Any,
):
    fsum_known = compute_dtype(0.0)
    fsum_middle = compute_dtype(0.0)
    for l in range(_q):
        if missing_mask[_opp_indices[l]] == wp.uint8(1):
            fsum_known += compute_dtype(2.0) * fpop[l]
        elif missing_mask[l] != wp.uint8(1):
            fsum_middle += fpop[l]
    return fsum_known + fsum_middle


@wp.func
def get_normal_vectors(
    missing_mask: Any,
):
    if wp.static(_d == 3):
        for l in range(_q):
            if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) + wp.abs(_c[2, l]) == 1:
                return -_u_vec(_c_float[0, l], _c_float[1, l], _c_float[2, l])
    else:
        for l in range(_q):
            if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) == 1:
                return -_u_vec(_c_float[0, l], _c_float[1, l])


@wp.func
def bounceback_nonequilibrium(
    fpop: Any,
    feq: Any,
    missing_mask: Any,
):
    for l in range(_q):
        if missing_mask[l] == wp.uint8(1):
            fpop[l] = fpop[_opp_indices[l]] + feq[l] - feq[_opp_indices[l]]
    return fpop


@wp.func
def regularize_fpop(
    fpop: Any,
    feq: Any,
):
    """
    Regularizes the distribution functions by adding non-equilibrium contributions based on second moments of fpop.
    """
    # Compute momentum flux of off-equilibrium populations for regularization: Pi^1 = Pi^{neq}
    f_neq = fpop - feq
    PiNeq = momentum_flux.warp_functional(f_neq)

    # Compute double dot product Qi:Pi1 (where Pi1 = PiNeq)
    nt = _d * (_d + 1) // 2
    for l in range(_q):
        QiPi1 = compute_dtype(0.0)
        for t in range(nt):
            QiPi1 += _qi[l, t] * PiNeq[t]

        # assign all populations based on eq 45 of Latt et al (2008)
        # fneq ~ f^1
        fpop1 = compute_dtype(4.5) * _w[l] * QiPi1
        fpop[l] = feq[l] + fpop1
    return fpop

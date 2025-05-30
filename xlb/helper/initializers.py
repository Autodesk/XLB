from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.equilibrium import MultiresQuadraticEquilibrium


def initialize_eq(f, grid, velocity_set, precision_policy, compute_backend, rho=None, u=None):
    if rho is None:
        rho = grid.create_field(cardinality=1, fill_value=1.0, dtype=precision_policy.compute_precision)
    if u is None:
        u = grid.create_field(cardinality=velocity_set.d, fill_value=0.0, dtype=precision_policy.compute_precision)
    equilibrium = QuadraticEquilibrium()

    if compute_backend == ComputeBackend.JAX:
        f = equilibrium(rho, u)
    elif compute_backend == ComputeBackend.WARP:
        f = equilibrium(rho, u, f)
    elif compute_backend == ComputeBackend.NEON:
        f = equilibrium(rho, u, f)
    else:
        raise NotImplementedError(f"Backend {compute_backend} not implemented")

    del rho, u

    return f


def initialize_multires_eq(f, grid, velocity_set, precision_policy, backend, rho, u):
    equilibrium = MultiresQuadraticEquilibrium()
    equilibrium(rho, u, f, stream=0)
    return f

from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium


def initialize_eq(f, grid, velocity_set, backend, rho=None, u=None):
    rho = rho or grid.create_field(cardinality=1, fill_value=1.0)
    u = u or grid.create_field(cardinality=velocity_set.d, fill_value=0.0)
    equilibrium = QuadraticEquilibrium(compute_backend=backend)

    if backend == ComputeBackend.JAX:
        f = equilibrium(rho, u)

    elif backend == ComputeBackend.WARP:
        f = equilibrium(rho, u, f)

    del rho, u

    return f
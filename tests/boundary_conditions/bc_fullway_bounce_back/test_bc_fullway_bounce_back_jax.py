import pytest
import numpy as np
import jax.numpy as jnp
import xlb
import jax
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb import DefaultConfig
from xlb.operator.boundary_masker import IndicesBoundaryMasker


def init_xlb_env(velocity_set):
    vel_set = velocity_set(precision_policy=xlb.PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.JAX)
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
        default_backend=ComputeBackend.JAX,
        velocity_set=vel_set,
    )


@pytest.mark.parametrize(
    "dim,velocity_set,grid_shape",
    [
        (2, xlb.velocity_set.D2Q9, (50, 50)),
        (2, xlb.velocity_set.D2Q9, (100, 100)),
        (3, xlb.velocity_set.D3Q19, (50, 50, 50)),
        (3, xlb.velocity_set.D3Q19, (100, 100, 100)),
        (3, xlb.velocity_set.D3Q27, (50, 50, 50)),
        (3, xlb.velocity_set.D3Q27, (100, 100, 100)),
    ],
)
def test_fullway_bounce_back_jax(dim, velocity_set, grid_shape):
    init_xlb_env(velocity_set)
    my_grid = grid_factory(grid_shape)
    velocity_set = DefaultConfig.velocity_set

    missing_mask = my_grid.create_field(cardinality=velocity_set.q, dtype=xlb.Precision.BOOL)

    bc_mask = my_grid.create_field(cardinality=1, dtype=xlb.Precision.UINT8)

    indices_boundary_masker = IndicesBoundaryMasker()

    # Make indices for boundary conditions (sphere)
    sphere_radius = grid_shape[0] // 4
    nr = grid_shape[0]
    x = np.arange(nr)
    y = np.arange(nr)
    z = np.arange(nr)
    if dim == 2:
        X, Y = np.meshgrid(x, y)
        indices = np.where((X - nr // 2) ** 2 + (Y - nr // 2) ** 2 < sphere_radius**2)
    else:
        X, Y, Z = np.meshgrid(x, y, z)
        indices = np.where((X - nr // 2) ** 2 + (Y - nr // 2) ** 2 + (Z - nr // 2) ** 2 < sphere_radius**2)

    indices = [tuple(indices[i]) for i in range(velocity_set.d)]
    fullway_bc = xlb.operator.boundary_condition.FullwayBounceBackBC(indices=indices)

    bc_mask, missing_mask = indices_boundary_masker([fullway_bc], bc_mask, missing_mask, start_index=None)

    f_pre = my_grid.create_field(cardinality=velocity_set.q, dtype=xlb.Precision.FP32, fill_value=0.0)
    # Generate a random field with the same shape
    key = jax.random.PRNGKey(0)
    random_field = jax.random.uniform(key, f_pre.shape)
    # Add the random field to f_pre
    f_pre += random_field

    f_post = my_grid.create_field(
        cardinality=velocity_set.q, dtype=xlb.Precision.FP32, fill_value=2.0
    )  # Arbitrary value so that we can check if the values are changed outside the boundary

    f = fullway_bc(f_pre, f_post, bc_mask, missing_mask)

    assert f.shape == (velocity_set.q,) + grid_shape

    for i in range(velocity_set.q):
        jnp.allclose(
            f[velocity_set.opp_indices[i]][tuple(indices)],
            f_pre[i][tuple(indices)],
        )

    # Make sure that everywhere else the values are the same as f_post. Note that indices are just int values
    mask_outside = np.ones(grid_shape, dtype=bool)
    mask_outside[indices] = False  # Mark boundary as false
    if dim == 2:
        for i in range(velocity_set.q):
            assert jnp.allclose(f[i, mask_outside], f_post[i, mask_outside])
    else:
        for i in range(velocity_set.q):
            assert jnp.allclose(f[i, mask_outside], f_post[i, mask_outside])

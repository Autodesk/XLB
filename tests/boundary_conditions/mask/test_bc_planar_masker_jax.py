import pytest
import jax.numpy as jnp
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.default_config import DefaultConfig
from xlb.grid import grid_factory


def init_xlb_env(velocity_set):
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
        default_backend=ComputeBackend.JAX,
        velocity_set=velocity_set,
    )


@pytest.mark.parametrize(
    "dim,velocity_set,grid_shape",
    [
        (2, xlb.velocity_set.D2Q9, (4, 4)),
        (2, xlb.velocity_set.D2Q9, (50, 50)),
        (2, xlb.velocity_set.D2Q9, (100, 100)),
        (3, xlb.velocity_set.D3Q19, (50, 50, 50)),
        (3, xlb.velocity_set.D3Q19, (100, 100, 100)),
        (3, xlb.velocity_set.D3Q27, (50, 50, 50)),
        (3, xlb.velocity_set.D3Q27, (100, 100, 100)),
    ],
)
def test_planar_masker_jax(dim, velocity_set, grid_shape):
    init_xlb_env(velocity_set)
    my_grid = grid_factory(grid_shape)
    velocity_set = DefaultConfig.velocity_set

    missing_mask = my_grid.create_field(
        cardinality=velocity_set.q, dtype=xlb.Precision.BOOL
    )

    fill_value = 0
    boundary_id_field = my_grid.create_field(
        cardinality=1, dtype=xlb.Precision.UINT8, fill_value=0
    )

    planar_boundary_masker = xlb.operator.boundary_masker.PlanarBoundaryMasker()

    if dim == 2:
        lower_bound = (0, 0)
        upper_bound = (1, grid_shape[1])
        direction = (1, 0)
    else:  # dim == 3
        lower_bound = (0, 0, 0)
        upper_bound = (1, grid_shape[1], grid_shape[2])
        direction = (1, 0, 0)

    start_index = (0,) * dim
    id_number = 1

    boundary_id_field, missing_mask = planar_boundary_masker(
        lower_bound,
        upper_bound,
        direction,
        id_number,
        boundary_id_field,
        missing_mask,
        start_index,
    )

    # Assert that the boundary condition is set on the left side of the domain based on the lower and upper bounds
    expected_slice = (slice(None),) + tuple(
        slice(lb, ub) for lb, ub in zip(lower_bound, upper_bound)
    )
    assert jnp.all(
        boundary_id_field[expected_slice] == id_number
    ), "Boundary not set correctly"

    # Assert that the rest of the domain is not affected and is equal to fill_value
    full_slice = tuple(slice(None) for _ in grid_shape)
    mask = jnp.ones_like(boundary_id_field, dtype=bool)
    mask = mask.at[expected_slice].set(False)
    assert jnp.all(
        boundary_id_field[full_slice][mask] == fill_value
    ), "Rest of domain incorrectly affected"


if __name__ == "__main__":
    pytest.main()

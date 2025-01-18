import pytest
import jax
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from jax.sharding import Mesh
from jax.experimental import mesh_utils
import jax.numpy as jnp


def init_xlb_env(velocity_set):
    vel_set = velocity_set(precision_policy=xlb.PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.WARP)
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
        default_backend=ComputeBackend.JAX,
        velocity_set=vel_set,
    )


@pytest.mark.parametrize("grid_size", [50, 100, 150])
def test_jax_2d_grid_initialization(grid_size):
    init_xlb_env(xlb.velocity_set.D2Q9)
    grid_shape = (grid_size, grid_size)
    my_grid = grid_factory(grid_shape)
    f = my_grid.create_field(cardinality=9)
    n_devices = jax.device_count()

    device_mesh = mesh_utils.create_device_mesh((1, n_devices, 1))
    expected_mesh = Mesh(device_mesh, axis_names=("cardinality", "x", "y"))

    assert f.shape == (9,) + grid_shape, "Field shape is incorrect"
    assert f.sharding.mesh == expected_mesh, "Field sharding mesh is incorrect"
    assert f.sharding.spec == ("cardinality", "x", "y"), "PartitionSpec is incorrect"


@pytest.mark.parametrize("grid_size", [50, 100, 150])
def test_jax_3d_grid_initialization(grid_size):
    init_xlb_env(xlb.velocity_set.D3Q19)
    grid_shape = (grid_size, grid_size, grid_size)
    my_grid = grid_factory(grid_shape)
    f = my_grid.create_field(cardinality=9)
    n_devices = jax.device_count()

    device_mesh = mesh_utils.create_device_mesh((1, n_devices, 1, 1))
    expected_mesh = Mesh(device_mesh, axis_names=("cardinality", "x", "y", "z"))

    assert f.shape == (9,) + grid_shape, "Field shape is incorrect"
    assert f.sharding.mesh == expected_mesh, "Field sharding mesh is incorrect"
    assert f.sharding.spec == (
        "cardinality",
        "x",
        "y",
        "z",
    ), "PartitionSpec is incorrect"


def test_jax_grid_create_field_fill_value():
    init_xlb_env(xlb.velocity_set.D2Q9)
    grid_shape = (100, 100)
    fill_value = 3.14
    my_grid = grid_factory(grid_shape)

    f = my_grid.create_field(cardinality=9, fill_value=fill_value)
    assert f.shape == (9,) + grid_shape, "Field shape is incorrect"
    assert jnp.allclose(f, fill_value), "Field not properly initialized with fill_value"


@pytest.fixture(autouse=True)
def setup_xlb_env():
    init_xlb_env(xlb.velocity_set.D2Q9)


if __name__ == "__main__":
    pytest.main()

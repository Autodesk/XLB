import pytest
import jax.numpy as jnp
import numpy as np
import warp as wp
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.operator.stream import Stream
from xlb import DefaultConfig
from xlb.grid import grid_factory


def init_xlb_env(velocity_set):
    vel_set = velocity_set(precision_policy=xlb.PrecisionPolicy.FP32FP32, compute_backend=ComputeBackend.WARP)
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
        default_backend=ComputeBackend.WARP,
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
def test_stream_operator_warp(dim, velocity_set, grid_shape):
    init_xlb_env(velocity_set)
    my_grid_jax = grid_factory(grid_shape, compute_backend=ComputeBackend.JAX)
    velocity_set = DefaultConfig.velocity_set

    f_initial = my_grid_jax.create_field(cardinality=velocity_set.q)
    f_initial = f_initial.at[..., f_initial.shape[-1] // 2].set(1)

    expected = []

    if dim == 2:
        for i in range(velocity_set.q):
            expected.append(
                jnp.roll(
                    f_initial[i, ...],
                    (velocity_set.c[0][i], velocity_set.c[1][i]),
                    axis=(0, 1),
                )
            )
    elif dim == 3:
        for i in range(velocity_set.q):
            expected.append(
                jnp.roll(
                    f_initial[i, ...],
                    (velocity_set.c[0][i], velocity_set.c[1][i], velocity_set.c[2][i]),
                    axis=(0, 1, 2),
                )
            )

    expected = jnp.stack(expected, axis=0)

    if dim == 2:
        f_initial_warp = wp.array(f_initial[..., np.newaxis])

    elif dim == 3:
        f_initial_warp = wp.array(f_initial)

    stream_op = Stream()
    my_grid_warp = grid_factory(grid_shape, compute_backend=ComputeBackend.WARP)
    f_streamed = my_grid_warp.create_field(cardinality=velocity_set.q)
    f_streamed = stream_op(f_initial_warp, f_streamed)

    if len(grid_shape) == 2:
        assert jnp.allclose(f_streamed.numpy()[..., 0], np.array(expected)), "Streaming did not occur as expected"
    else:
        assert jnp.allclose(f_streamed.numpy(), np.array(expected)), "Streaming did not occur as expected"


if __name__ == "__main__":
    pytest.main()

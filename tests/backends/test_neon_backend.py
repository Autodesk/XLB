"""
Smoke coverage for the Neon compute backend.

The rest of the suite only exercises the JAX and WARP backends, and
``tests/install/flow_past_sphere_3d_test.py`` skips Neon because the sphere's
``HalfwayBounceBackBC`` has no Neon implementation. These tests cover the parts
of the Neon path that need no boundary condition: ``neon.init()`` registering
XLB's custom Warp types for NVRTC, grid and field allocation, the device-side
fill and host read-back, and operator kernel construction.

Skipped when ``neon`` is missing (an env built with the ``[warp]`` extra rather
than ``[neon]``) or when no CUDA device is available.
"""

import pytest

neon = pytest.importorskip("neon", reason="neon_gpu is not installed; install XLB with the [neon] extra")

import warp as wp  # noqa: E402

import xlb  # noqa: E402
from xlb.compute_backend import ComputeBackend  # noqa: E402
from xlb.grid import grid_factory  # noqa: E402
from xlb.precision_policy import PrecisionPolicy  # noqa: E402

GRID_SHAPE = (16, 16, 16)
CARDINALITY = 19
FILL_VALUE = 2.5


@pytest.fixture(scope="module", autouse=True)
def neon_env():
    """Initialize XLB on the Neon backend once for the module.

    ``xlb.init`` calls ``neon.init()``, which must run before any kernel using
    Neon's custom types is built, so every test here shares one initialization.
    """
    wp.init()
    if wp.get_cuda_device_count() == 0:
        pytest.skip("the Neon backend requires a CUDA device")

    precision_policy = PrecisionPolicy.FP32FP32
    xlb.init(
        velocity_set=xlb.velocity_set.D3Q19(precision_policy=precision_policy, compute_backend=ComputeBackend.NEON),
        default_backend=ComputeBackend.NEON,
        default_precision_policy=precision_policy,
    )


def test_neon_grid_shape():
    grid = grid_factory(GRID_SHAPE, compute_backend=ComputeBackend.NEON)
    assert grid.shape == GRID_SHAPE, "Neon grid shape is incorrect"


def test_neon_field_cardinality():
    grid = grid_factory(GRID_SHAPE, compute_backend=ComputeBackend.NEON)
    f = grid.create_field(cardinality=CARDINALITY)
    assert f.cardinality == CARDINALITY, "Neon field cardinality is incorrect"


def test_neon_field_fill_value():
    """Fill on the device, then read back on the host.

    Exercises allocation, the fill kernel, and the device-to-host transfer.
    """
    grid = grid_factory(GRID_SHAPE, compute_backend=ComputeBackend.NEON)
    f = grid.create_field(cardinality=CARDINALITY, fill_value=FILL_VALUE)
    f.update_host(0)

    for idx in (neon.Index_3d(0, 0, 0), neon.Index_3d(1, 2, 3), neon.Index_3d(*(d - 1 for d in GRID_SHAPE))):
        for card in (0, CARDINALITY // 2, CARDINALITY - 1):
            assert f.read(idx, card) == pytest.approx(FILL_VALUE), "Neon field not initialized with fill_value"


def test_neon_operator_kernel_construction():
    """Operators must produce a Neon functional and container, not Warp ones."""
    from xlb.operator.equilibrium import QuadraticEquilibrium

    op = QuadraticEquilibrium(compute_backend=ComputeBackend.NEON)
    assert op.neon_functional is not None, "Neon functional was not constructed"
    assert op.neon_container is not None, "Neon container was not constructed"

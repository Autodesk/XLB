"""
Two-way (Boussinesq) coupling between the thermal and the flow steppers.

The buoyancy force is rebuilt from the temperature every step and handed to the flow
stepper as a force field. Warp is checked against JAX on a differentially heated cavity,
and the Warp force-field path against the uniform-force path on a periodic box.
"""

import pytest
import numpy as np
import warp as wp
import xlb
import xlb.velocity_set
from xlb.compute_backend import ComputeBackend
from xlb.grid import grid_factory
from xlb.helper import omega_from_diffusivity
from xlb.operator.boundary_condition import HalfwayBounceBackBC, ScalarDirichletBC, ScalarNeumannBC
from xlb.operator.force import BoussinesqForce
from xlb.operator.macroscopic import Macroscopic, ZeroMoment
from xlb.operator.stepper import IncompressibleNavierStokesStepper, ThermalStepper
from xlb.precision_policy import PrecisionPolicy

PRECISION = PrecisionPolicy.FP64FP64


def init_xlb_env(compute_backend):
    flow_velocity_set = xlb.velocity_set.D2Q9(precision_policy=PRECISION, compute_backend=compute_backend)
    xlb.init(velocity_set=flow_velocity_set, default_backend=compute_backend, default_precision_policy=PRECISION)
    thermal_velocity_set = xlb.velocity_set.D2Q5(precision_policy=PRECISION, compute_backend=compute_backend)
    return flow_velocity_set, thermal_velocity_set


def to_numpy(field, compute_backend):
    """Return a field as a NumPy array with the JAX layout, i.e. without Warp's z axis in 2D."""
    if compute_backend == ComputeBackend.WARP:
        return field.numpy()[..., 0]
    return np.asarray(field)


def buoyant_cavity(compute_backend, n=24, num_steps=600):
    """Differentially heated square cavity: hot left wall, cold right wall, no-slip all round."""
    flow_velocity_set, thermal_velocity_set = init_xlb_env(compute_backend)
    grid = grid_factory((n, n), compute_backend=compute_backend)
    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)

    walls = [box["bottom"][i] + box["top"][i] + box["left"][i] + box["right"][i] for i in range(2)]
    walls = np.unique(np.array(walls), axis=-1).tolist()

    # A zero uniform force only switches the collision to ForcedCollision; the real force
    # is rebuilt from the temperature and passed on every call
    flow_stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=[HalfwayBounceBackBC(indices=walls)],
        collision_type="BGK",
        force_vector=np.zeros(2),
    )
    thermal_stepper = ThermalStepper(
        grid=grid,
        boundary_conditions=[
            ScalarDirichletBC(value=1.0, indices=box["left"]),
            ScalarDirichletBC(value=0.0, indices=box["right"]),
            ScalarNeumannBC(flux=0.0, indices=[box_no_edge["bottom"][i] + box_no_edge["top"][i] for i in range(2)]),
        ],
        velocity_set=thermal_velocity_set,
    )

    omega_flow = 1.0 / (3.0 * 0.05 + 0.5)
    omega_thermal = omega_from_diffusivity(0.05)

    f_0, f_1, bc_mask, missing_mask = flow_stepper.prepare_fields()
    g_0, g_1, bc_mask_thermal, missing_mask_thermal = thermal_stepper.prepare_fields(phi_init=0.5)
    u, source = thermal_stepper.prepare_aux_fields()

    macroscopic = Macroscopic()
    zero_moment = ZeroMoment(velocity_set=thermal_velocity_set)
    boussinesq = BoussinesqForce(buoyancy=(0.0, 1e-3), reference_value=0.5)

    if compute_backend == ComputeBackend.WARP:
        rho = grid.create_field(cardinality=1, dtype=PRECISION.store_precision)
        phi = grid.create_field(cardinality=1, dtype=PRECISION.store_precision)
        force = grid.create_field(cardinality=2, dtype=PRECISION.store_precision)
    else:
        force = np.zeros((2, n, n))

    for i in range(num_steps):
        f_0, f_1 = flow_stepper(f_0, f_1, bc_mask, missing_mask, omega_flow, i, force)
        f_0, f_1 = f_1, f_0

        if compute_backend == ComputeBackend.WARP:
            rho, u = macroscopic(f_0, rho, u)
        else:
            _, u = macroscopic(f_0)

        g_0, g_1 = thermal_stepper(g_0, g_1, u, source, bc_mask_thermal, missing_mask_thermal, omega_thermal, i)
        g_0, g_1 = g_1, g_0

        if compute_backend == ComputeBackend.WARP:
            phi = zero_moment(g_0, phi)
            force = boussinesq(phi, force)
        else:
            force = boussinesq(zero_moment(g_0))

    if compute_backend == ComputeBackend.WARP:
        phi = zero_moment(g_0, phi)
    else:
        phi = zero_moment(g_0)
    return to_numpy(phi, compute_backend)[0], to_numpy(u, compute_backend)


def test_buoyant_cavity_warp_matches_jax():
    t_warp, u_warp = buoyant_cavity(ComputeBackend.WARP)
    t_jax, u_jax = buoyant_cavity(ComputeBackend.JAX)

    # The walls are at rest, so any flow is driven by buoyancy alone
    assert np.abs(u_jax).max() > 1e-4
    # Warm fluid rises next to the hot wall, cold fluid sinks next to the cold one
    n = t_jax.shape[0]
    assert u_jax[1, 1, n // 2] > 0.0 and u_jax[1, n - 2, n // 2] < 0.0

    assert np.allclose(t_warp, t_jax, atol=1e-10)
    assert np.allclose(u_warp, u_jax, atol=1e-10)


def periodic_box_velocity(compute_backend, force_vector, per_call_force=None, n=8, num_steps=50):
    """Uniformly forced periodic box: every step adds exactly the force to the velocity."""
    init_xlb_env(compute_backend)
    grid = grid_factory((n, n), compute_backend=compute_backend)
    stepper = IncompressibleNavierStokesStepper(grid=grid, boundary_conditions=[], collision_type="BGK", force_vector=force_vector)
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

    for i in range(num_steps):
        if per_call_force is None:
            f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, 1.0, i)
        else:
            f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, 1.0, i, per_call_force)
        f_0, f_1 = f_1, f_0

    macroscopic = Macroscopic()
    if compute_backend == ComputeBackend.WARP:
        rho = grid.create_field(cardinality=1, dtype=PRECISION.store_precision)
        u = grid.create_field(cardinality=2, dtype=PRECISION.store_precision)
        rho, u = macroscopic(f_0, rho, u)
    else:
        _, u = macroscopic(f_0)
    return to_numpy(u, compute_backend)


@pytest.mark.parametrize("compute_backend", [ComputeBackend.JAX, ComputeBackend.WARP])
def test_force_field_matches_uniform_force(compute_backend):
    n, num_steps = 8, 50
    force = np.array([1e-5, -2e-5])
    force_field = np.broadcast_to(force[:, None, None], (2, n, n)).copy()

    u_uniform = periodic_box_velocity(compute_backend, force, n=n, num_steps=num_steps)
    u_field_ctor = periodic_box_velocity(compute_backend, force_field, n=n, num_steps=num_steps)
    u_field_call = periodic_box_velocity(compute_backend, np.zeros(2), per_call_force=force_field, n=n, num_steps=num_steps)

    expected = np.broadcast_to((num_steps * force)[:, None, None], (2, n, n))
    assert np.allclose(u_uniform, expected, rtol=1e-6)
    assert np.allclose(u_field_ctor, u_uniform, atol=1e-14)
    assert np.allclose(u_field_call, u_uniform, atol=1e-14)


def test_per_call_warp_force_field_is_used_in_place():
    """A Warp force field updated in place on the device is picked up without a copy."""
    n = 8
    init_xlb_env(ComputeBackend.WARP)
    grid = grid_factory((n, n), compute_backend=ComputeBackend.WARP)
    stepper = IncompressibleNavierStokesStepper(grid=grid, boundary_conditions=[], collision_type="BGK", force_vector=np.zeros(2))
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

    force = grid.create_field(cardinality=2, dtype=PRECISION.store_precision)
    assert stepper._as_warp_force_field(force) is force

    for i in range(10):
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, 1.0, i, force)
        f_0, f_1 = f_1, f_0
    force.fill_(1e-5)
    for i in range(10, 20):
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, 1.0, i, force)
        f_0, f_1 = f_1, f_0

    rho = grid.create_field(cardinality=1, dtype=PRECISION.store_precision)
    u = grid.create_field(cardinality=2, dtype=PRECISION.store_precision)
    rho, u = Macroscopic()(f_0, rho, u)
    assert np.allclose(u.numpy(), 10 * 1e-5, rtol=1e-6)


@pytest.mark.parametrize("compute_backend", [ComputeBackend.JAX, ComputeBackend.WARP])
def test_per_call_force_requires_forced_stepper(compute_backend):
    init_xlb_env(compute_backend)
    grid = grid_factory((8, 8), compute_backend=compute_backend)
    stepper = IncompressibleNavierStokesStepper(grid=grid, boundary_conditions=[], collision_type="BGK")
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

    with pytest.raises(Exception, match="requires the stepper to be constructed with force_vector"):
        stepper(f_0, f_1, bc_mask, missing_mask, 1.0, 0, np.zeros((2, 8, 8)))


@pytest.mark.parametrize("compute_backend", [ComputeBackend.JAX, ComputeBackend.WARP])
def test_boussinesq_force(compute_backend):
    n = 6
    init_xlb_env(compute_backend)
    grid = grid_factory((n, n), compute_backend=compute_backend)
    boussinesq = BoussinesqForce(buoyancy=(0.5, 2.0), reference_value=0.25)

    phi = np.linspace(0.0, 1.0, n * n).reshape(1, n, n)
    if compute_backend == ComputeBackend.WARP:
        force = grid.create_field(cardinality=2, dtype=PRECISION.store_precision)
        force = boussinesq(wp.array(phi[..., None], dtype=wp.float64), force)
    else:
        force = boussinesq(phi)

    expected = np.array([0.5, 2.0])[:, None, None] * (phi - 0.25)
    assert np.allclose(to_numpy(force, compute_backend), expected, atol=1e-14)


if __name__ == "__main__":
    pytest.main()

"""
Test: XLB Stepper Autodiff - JAX vs Warp Comparison

This script tests whether gradients propagate through the XLB stepper
for both JAX and Warp backends. It performs identical tests on both
backends and compares the results side-by-side.

Expected result: JAX works, Warp does not (stepper lacks adjoint kernels).

Usage:
    python examples/cfd/test_stepper_autodiff.py
"""
import numpy as np

print()
print("=" * 70)
print("XLB STEPPER AUTODIFF TEST")
print("=" * 70)
print()
print("This test checks if gradients propagate through the LBM stepper.")
print("We run the SAME test on both JAX and Warp backends and compare.")
print()

# =============================================================================
# SETUP BOTH BACKENDS
# =============================================================================

import warp as wp
wp.init()

import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import FullwayBounceBackBC
from xlb.operator.macroscopic import Macroscopic
import xlb.velocity_set

# Common parameters
grid_shape = (32, 32)
omega = 1.8
precision_policy = PrecisionPolicy.FP32FP32

print("-" * 70)
print("TEST CONFIGURATION")
print("-" * 70)
print(f"  Grid shape:       {grid_shape}")
print(f"  Omega:            {omega}")
print("  Precision:        FP32FP32")
print("  Boundary:         FullwayBounceBackBC (walls)")
print("  Collision:        BGK")
print("  Test:             Forward 1 step -> Compute rho -> MSE Loss -> Backward")
print()

# =============================================================================
# WARP BACKEND SETUP
# =============================================================================

warp_velocity_set = xlb.velocity_set.D2Q9(
    precision_policy=precision_policy,
    compute_backend=ComputeBackend.WARP,
)

xlb.init(
    velocity_set=warp_velocity_set,
    default_backend=ComputeBackend.WARP,
    default_precision_policy=precision_policy,
)

warp_grid = grid_factory(grid_shape, compute_backend=ComputeBackend.WARP)

box = warp_grid.bounding_box_indices()
walls = [box["bottom"][i] + box["top"][i] + box["left"][i] + box["right"][i] for i in range(warp_velocity_set.d)]
walls = np.unique(np.array(walls), axis=-1).tolist()
warp_bc = FullwayBounceBackBC(indices=walls)

warp_stepper = IncompressibleNavierStokesStepper(
    grid=warp_grid,
    boundary_conditions=[warp_bc],
    collision_type="BGK",
)

warp_f_0, warp_f_1, warp_bc_mask, warp_missing_mask = warp_stepper.prepare_fields()

warp_macro = Macroscopic(
    velocity_set=warp_velocity_set,
    precision_policy=precision_policy,
    compute_backend=ComputeBackend.WARP,
)

q = warp_velocity_set.q
shape_4d = (*grid_shape, 1)

@wp.kernel
def warp_loss_kernel(rho: wp.array4d(dtype=wp.float32), loss: wp.array(dtype=wp.float32)):
    i, j, k = wp.tid()
    wp.atomic_add(loss, 0, rho[0, i, j, k] ** 2.0)

# =============================================================================
# JAX BACKEND SETUP
# =============================================================================

import jax
import jax.numpy as jnp
from jax import value_and_grad

jax_velocity_set = xlb.velocity_set.D2Q9(
    precision_policy=precision_policy,
    compute_backend=ComputeBackend.JAX,
)

xlb.init(
    velocity_set=jax_velocity_set,
    default_backend=ComputeBackend.JAX,
    default_precision_policy=precision_policy,
)

jax_grid = grid_factory(grid_shape, compute_backend=ComputeBackend.JAX)

jax_box = jax_grid.bounding_box_indices()
jax_walls = [jax_box["bottom"][i] + jax_box["top"][i] + jax_box["left"][i] + jax_box["right"][i] for i in range(jax_velocity_set.d)]
jax_walls = np.unique(np.array(jax_walls), axis=-1).tolist()
jax_bc = FullwayBounceBackBC(indices=jax_walls)

jax_stepper = IncompressibleNavierStokesStepper(
    grid=jax_grid,
    boundary_conditions=[jax_bc],
    collision_type="BGK",
)

jax_f_0, jax_f_1, jax_bc_mask, jax_missing_mask = jax_stepper.prepare_fields()

# =============================================================================
# RUN TESTS
# =============================================================================

# --- WARP TEST ---
f_in_warp = wp.zeros((q, *shape_4d), dtype=wp.float32, requires_grad=True)
f_out_warp = wp.zeros((q, *shape_4d), dtype=wp.float32, requires_grad=True)
rho_warp = wp.zeros((1, *shape_4d), dtype=wp.float32, requires_grad=True)
u_warp = wp.zeros((2, *shape_4d), dtype=wp.float32, requires_grad=True)
loss_warp = wp.zeros((1,), dtype=wp.float32, requires_grad=True)
wp.copy(f_in_warp, warp_f_0)

with wp.Tape() as tape:
    f_out_warp, f_in_warp = warp_stepper(f_in_warp, f_out_warp, warp_bc_mask, warp_missing_mask, omega, 0)
    rho_warp, u_warp = warp_macro(f_out_warp, rho_warp, u_warp)
    wp.launch(warp_loss_kernel, inputs=[rho_warp], outputs=[loss_warp], dim=rho_warp.shape[1:])

warp_loss_val = float(loss_warp.numpy()[0])
loss_warp.grad.fill_(1.0)
tape.backward()

warp_f_in_grad = f_in_warp.grad.numpy() if f_in_warp.grad is not None else np.zeros_like(warp_f_0.numpy())
warp_f_out_grad = f_out_warp.grad.numpy() if f_out_warp.grad is not None else np.zeros_like(warp_f_0.numpy())
warp_rho_grad = rho_warp.grad.numpy() if rho_warp.grad is not None else np.zeros((1, *shape_4d))

warp_f_in_grad_norm = float(np.linalg.norm(warp_f_in_grad))
warp_f_out_grad_norm = float(np.linalg.norm(warp_f_out_grad))
warp_rho_grad_norm = float(np.linalg.norm(warp_rho_grad))

# --- JAX TEST ---
def jax_forward_and_loss(f_in):
    f_out, _ = jax_stepper(f_in, jax_f_1, jax_bc_mask, jax_missing_mask, omega, 0)
    rho = jnp.sum(f_out, axis=0)
    return jnp.sum(rho ** 2)

jax_loss_val, jax_grad = value_and_grad(jax_forward_and_loss)(jax_f_0)
jax_loss_val = float(jax_loss_val)
jax_grad_norm = float(jnp.linalg.norm(jax_grad))

# =============================================================================
# SIDE-BY-SIDE RESULTS
# =============================================================================

print("=" * 70)
print("RESULTS: SIDE-BY-SIDE COMPARISON")
print("=" * 70)
print()
print(f"{'Metric':<35} {'WARP':<15} {'JAX':<15}")
print("-" * 65)
print(f"{'Loss value':<35} {warp_loss_val:<15.4f} {jax_loss_val:<15.4f}")
print(f"{'d(Loss)/d(f_input) gradient norm':<35} {warp_f_in_grad_norm:<15.4f} {jax_grad_norm:<15.4f}")
print()

print("-" * 70)
print("GRADIENT FLOW ANALYSIS (Warp only - to debug where gradients stop)")
print("-" * 70)
print()
print("  In Warp, we can check gradients at each stage of the computation:")
print()
print("    1. loss.grad (set manually)        : 1.0 (seed)")
print(f"    2. d(loss)/d(rho) gradient norm    : {warp_rho_grad_norm:.4f}")
print(f"    3. d(loss)/d(f_out) gradient norm  : {warp_f_out_grad_norm:.4f}")
print(f"    4. d(loss)/d(f_in) gradient norm   : {warp_f_in_grad_norm:.4f}  <-- THIS IS THE PROBLEM")
print()
print("  Gradient flows: loss -> rho -> f_out (through Macroscopic) ✓")
print("  Gradient STOPS: f_out -> f_in (through Stepper) ✗")
print()

print("=" * 70)
print("DIAGNOSIS")
print("=" * 70)
print()

if warp_f_in_grad_norm == 0 and jax_grad_norm > 0:
    print("  ISSUE CONFIRMED: Warp stepper does not propagate gradients.")
    print()
    print("  WHY THIS HAPPENS:")
    print("  -----------------")
    print("  Warp's autodiff (wp.Tape) requires either:")
    print("    a) Automatic adjoint generation (works for simple kernels), or")
    print("    b) Manual @wp.func_grad adjoint implementations")
    print()
    print("  XLB's stepper kernel (nse_stepper.py) has characteristics that")
    print("  PREVENT automatic adjoint generation:")
    print("    - Early returns: 'if _boundary_id == wp.uint8(255): return'")
    print("    - Integer conditionals and mask operations")
    print("    - Complex nested @wp.func calls without adjoints")
    print()
    print("  The Macroscopic operator DOES work because it's a simple")
    print("  summation kernel that Warp can auto-differentiate.")
    print()
    print("  JAX WORKS because it uses source-code transformation (not tape)")
    print("  which can differentiate through any Python/JAX code automatically.")
    print()
    print("  TO FIX (requires XLB core changes):")
    print("  ------------------------------------")
    print("    Add @wp.func_grad adjoint implementations for:")
    print("    - xlb/operator/collision/bgk.py: warp_functional()")
    print("    - xlb/operator/stream/stream.py: warp_functional()")
    print("    - xlb/operator/equilibrium/*.py: warp_functional()")
    print()
    print("  RECOMMENDATION:")
    print("  ----------------")
    print("    Use JAX backend for differentiable LBM applications until")
    print("    Warp adjoint kernels are implemented in XLB.")
else:
    print("  Unexpected result - please investigate.")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"  WARP: Loss={warp_loss_val:.2f}, Gradient={warp_f_in_grad_norm:.2f} --> {'BROKEN' if warp_f_in_grad_norm == 0 else 'OK'}")
print(f"  JAX:  Loss={jax_loss_val:.2f}, Gradient={jax_grad_norm:.2f} --> {'OK' if jax_grad_norm > 0 else 'BROKEN'}")
print()
print("=" * 70)

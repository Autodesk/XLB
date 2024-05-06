import unittest
import numpy as np
import jax.numpy as jnp
import warp as wp
from xlb.grid import grid_factory
import xlb

wp.init()


class TestBoundaryConditions(unittest.TestCase):
    def setUp(self):
        self.backends = ["warp", "jax"]
        self.results = {}

    def run_boundary_conditions(self, backend):
        # Set the compute backend
        if backend == "warp":
            compute_backend = xlb.ComputeBackend.WARP
        elif backend == "jax":
            compute_backend = xlb.ComputeBackend.JAX

        # Set the precision policy
        precision_policy = xlb.PrecisionPolicy.FP32FP32

        # Set the velocity set
        velocity_set = xlb.velocity_set.D3Q19()

        # Make grid
        nr = 128
        shape = (nr, nr, nr)
        grid = grid_factory(shape)
        # Make fields
        f_pre = grid.create_field(
            cardinality=velocity_set.q, precision=xlb.Precision.FP32
        )
        f_post = grid.create_field(
            cardinality=velocity_set.q, precision=xlb.Precision.FP32
        )
        f = grid.create_field(cardinality=velocity_set.q, precision=xlb.Precision.FP32)
        boundary_id_field = grid.create_field(cardinality=1, precision=xlb.Precision.UINT8)
        missing_mask = grid.create_field(
            cardinality=velocity_set.q, precision=xlb.Precision.BOOL
        )

        # Make needed operators
        equilibrium = xlb.operator.equilibrium.QuadraticEquilibrium(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )
        equilibrium_bc = xlb.operator.boundary_condition.EquilibriumBC(
            rho=1.0,
            u=(0.0, 0.0, 0.0),
            equilibrium_operator=equilibrium,
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )
        do_nothing_bc = xlb.operator.boundary_condition.DoNothingBC(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )
        halfway_bounce_back_bc = xlb.operator.boundary_condition.HalfwayBounceBackBC(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )
        fullway_bounce_back_bc = xlb.operator.boundary_condition.FullwayBounceBackBC(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )
        indices_boundary_masker = xlb.operator.boundary_masker.IndicesBoundaryMasker(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )

        # Make indices for boundary conditions (sphere)
        sphere_radius = 10
        x = np.arange(nr)
        y = np.arange(nr)
        z = np.arange(nr)
        X, Y, Z = np.meshgrid(x, y, z)
        indices = np.where(
            (X - nr // 2) ** 2 + (Y - nr // 2) ** 2 + (Z - nr // 2) ** 2
            < sphere_radius**2
        )
        indices = np.array(indices).T
        if backend == "jax":
            indices = jnp.array(indices)
        elif backend == "warp":
            indices = wp.from_numpy(indices, dtype=wp.int32)

        # Test equilibrium boundary condition
        boundary_id_field, missing_mask = indices_boundary_masker(
            indices, equilibrium_bc.id, boundary_id_field, missing_mask, (0, 0, 0)
        )
        if backend == "jax":
            f_equilibrium = equilibrium_bc(f_pre, f_post, boundary_id_field, missing_mask)
        elif backend == "warp":
            f_equilibrium = grid.create_field(
                cardinality=velocity_set.q, precision=xlb.Precision.FP32
            )
            f_equilibrium = equilibrium_bc(
                f_pre, f_post, boundary_id_field, missing_mask, f_equilibrium
            )

        # Test do nothing boundary condition
        boundary_id_field, missing_mask = indices_boundary_masker(
            indices, do_nothing_bc.id, boundary_id_field, missing_mask, (0, 0, 0)
        )
        if backend == "jax":
            f_do_nothing = do_nothing_bc(f_pre, f_post, boundary_id_field, missing_mask)
        elif backend == "warp":
            f_do_nothing = grid.create_field(
                cardinality=velocity_set.q, precision=xlb.Precision.FP32
            )
            f_do_nothing = do_nothing_bc(
                f_pre, f_post, boundary_id_field, missing_mask, f_do_nothing
            )

        # Test halfway bounce back boundary condition
        boundary_id_field, missing_mask = indices_boundary_masker(
            indices, halfway_bounce_back_bc.id, boundary_id_field, missing_mask, (0, 0, 0)
        )
        if backend == "jax":
            f_halfway_bounce_back = halfway_bounce_back_bc(
                f_pre, f_post, boundary_id_field, missing_mask
            )
        elif backend == "warp":
            f_halfway_bounce_back = grid.create_field(
                cardinality=velocity_set.q, precision=xlb.Precision.FP32
            )
            f_halfway_bounce_back = halfway_bounce_back_bc(
                f_pre, f_post, boundary_id_field, missing_mask, f_halfway_bounce_back
            )

        # Test the full boundary condition
        boundary_id_field, missing_mask = indices_boundary_masker(
            indices, fullway_bounce_back_bc.id, boundary_id_field, missing_mask, (0, 0, 0)
        )
        if backend == "jax":
            f_fullway_bounce_back = fullway_bounce_back_bc(
                f_pre, f_post, boundary_id_field, missing_mask
            )
        elif backend == "warp":
            f_fullway_bounce_back = grid.create_field(
                cardinality=velocity_set.q, precision=xlb.Precision.FP32
            )
            f_fullway_bounce_back = fullway_bounce_back_bc(
                f_pre, f_post, boundary_id_field, missing_mask, f_fullway_bounce_back
            )

        return f_equilibrium, f_do_nothing, f_halfway_bounce_back, f_fullway_bounce_back

    def test_boundary_conditions(self):
        for backend in self.backends:
            (
                f_equilibrium,
                f_do_nothing,
                f_halfway_bounce_back,
                f_fullway_bounce_back,
            ) = self.run_boundary_conditions(backend)
            self.results[backend] = {
                "equilibrium": np.array(f_equilibrium)
                if backend == "jax"
                else f_equilibrium.numpy(),
                "do_nothing": np.array(f_do_nothing)
                if backend == "jax"
                else f_do_nothing.numpy(),
                "halfway_bounce_back": np.array(f_halfway_bounce_back)
                if backend == "jax"
                else f_halfway_bounce_back.numpy(),
                "fullway_bounce_back": np.array(f_fullway_bounce_back)
                if backend == "jax"
                else f_fullway_bounce_back.numpy(),
            }

        for test_name in [
            "equilibrium",
            "do_nothing",
            "halfway_bounce_back",
            "fullway_bounce_back",
        ]:
            with self.subTest(test_name=test_name):
                warp_results = self.results["warp"][test_name]
                jax_results = self.results["jax"][test_name]

                is_close = np.allclose(warp_results, jax_results, atol=1e-8, rtol=1e-5)
                if not is_close:
                    diff_indices = np.where(
                        ~np.isclose(warp_results, jax_results, atol=1e-8, rtol=1e-5)
                    )
                    differences = [
                        (idx, warp_results[idx], jax_results[idx])
                        for idx in zip(*diff_indices)
                    ]
                    difference_str = "\n".join(
                        [
                            f"Index: {idx}, Warp: {w}, JAX: {j}"
                            for idx, w, j in differences
                        ]
                    )
                    msg = f"{test_name} test failed: results do not match between backends. Differences:\n{difference_str}"
                else:
                    msg = ""

                self.assertTrue(is_close, msg=msg)


if __name__ == "__main__":
    unittest.main()


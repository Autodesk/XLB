from typing import Any
import warp as wp

class L2Loss:

    @wp.kernel
    def _l2_loss(
        rho: wp.array4d(dtype=Any),
        target_rho: wp.array4d(dtype=Any),
        boundary_id: wp.array4d(dtype=wp.uint8),
        l2_loss: wp.array(dtype=wp.float32),
    ):
        # Get the global index
        i, j, k = wp.tid()

        # Compute the loss
        if boundary_id[0, i, j, k] == wp.uint8(0):
            wp.atomic_add(l2_loss, 0, (rho[0, i, j, k] - target_rho[0, i, j, k])**2.0)

    def __call__(self, rho, target_rho, boundary_id, l2_loss):

        # Launch the warp kernel
        wp.launch(
            self._l2_loss,
            inputs=[
                rho,
                target_rho,
                boundary_id,
            ],
            outputs=[
                l2_loss
            ],
            dim=rho.shape[1:],
        )

        return l2_loss


from typing import Any
import warp as wp

class GradientDescent:
    """
    Gradient descent operator.
    """

    @wp.kernel
    def gradient_decent(
        field: wp.array4d(dtype=Any),
        adj_field: wp.array4d(dtype=Any),
        learning_rate: wp.float32,
    ):
        # Get the global index
        i, j, k = wp.tid()

        # Update the field
        for ii in range(field.shape[0]):
            field[ii, i, j, k] -= learning_rate * adj_field[ii, i, j, k]

    def __call__(
        self,
        field: wp.array4d(dtype=Any),
        adj_field: wp.array4d(dtype=Any),
        learning_rate: float,
    ):

        # Launch the warp kernel
        wp.launch(
            self.gradient_decent,
            inputs=[
                field,
                adj_field,
                learning_rate,
            ],
            dim=field.shape[1:],
        )
        return field

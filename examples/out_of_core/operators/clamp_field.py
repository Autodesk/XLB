from typing import Any
import warp as wp

class ClampField:
    """
    Clamp field operator.
    """

    @wp.kernel
    def clamp_field(
        field: wp.array4d(dtype=Any),
        min_val: wp.array(dtype=Any),
        max_val: wp.array(dtype=Any),
    ):
        # Get the global index
        i, j, k = wp.tid()

        # Update the field
        for ii in range(field.shape[0]):
            field[ii, i, j, k] = wp.max(min_val[ii], wp.min(max_val[ii], field[ii, i, j, k]))

    def __call__(
        self,
        field: wp.array4d(dtype=Any),
        min_val: wp.array(dtype=Any),
        max_val: wp.array(dtype=Any),
    ):

        # Launch the warp kernel
        wp.launch(
            self.clamp_field,
            inputs=[
                field,
                min_val,
                max_val,
            ],
            dim=field.shape[1:],
        )
        return field


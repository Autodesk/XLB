from xlb.operator.boundary_masker import PlanarBoundaryMasker


def assign_bc_id_box_faces(boundary_mask, missing_mask, shape, bc_id, sides, backend=None):
    """
    Assign boundary conditions for specified sides of 2D and 3D boxes using planar_boundary_masker function.

    Parameters:
    boundary_mask: ndarray
        The field containing boundary IDs.
    missing_mask: ndarray
        The mask indicating missing boundary IDs.
    shape: tuple
        The shape of the grid (extent of the grid in each dimension).
    bc_id: int
        The boundary condition ID to assign to the specified boundaries.
    sides: list of str
        The list of sides to apply conditions to. Valid values for 2D are 'bottom', 'top', 'left', 'right'.
        Valid values for 3D are 'bottom', 'top', 'front', 'back', 'left', 'right'.
    """

    planar_boundary_masker = PlanarBoundaryMasker(compute_backend=backend)

    def apply(lower_bound, upper_bound, direction, reference=(0, 0, 0)):
        nonlocal boundary_mask, missing_mask, planar_boundary_masker
        boundary_mask, missing_mask = planar_boundary_masker(
            lower_bound,
            upper_bound,
            direction,
            bc_id,
            boundary_mask,
            missing_mask,
            reference,
        )

    dimensions = len(shape)

    if dimensions == 2:
        nr, nc = shape
        for boundary in sides:
            if boundary == "bottom":
                apply((0, 0), (nr, 1), (1, 0), (0, 0))
            elif boundary == "top":
                apply((0, nc - 1), (nr, nc), (1, 0), (0, 0))
            elif boundary == "left":
                apply((0, 0), (1, nc), (0, 1), (0, 0))
            elif boundary == "right":
                apply((nr - 1, 0), (nr, nc), (0, 1), (0, 0))

    elif dimensions == 3:
        nr, nc, nz = shape
        for boundary in sides:
            if boundary == "bottom":
                apply((0, 0, 0), (nr, 1, nz), (1, 0, 0), (0, 0, 0))
            elif boundary == "top":
                apply((0, nc - 1, 0), (nr, nc, nz), (1, 0, 0), (0, 0, 0))
            elif boundary == "front":
                apply((0, 0, 0), (nr, nc, 1), (0, 1, 0), (0, 0, 0))
            elif boundary == "back":
                apply((0, 0, nz - 1), (nr, nc, nz), (0, 1, 0), (0, 0, 0))
            elif boundary == "left":
                apply((0, 0, 0), (1, nc, nz), (0, 0, 1), (0, 0, 0))
            elif boundary == "right":
                apply((nr - 1, 0, 0), (nr, nc, nz), (0, 0, 1), (0, 0, 0))

    else:
        raise ValueError("Unsupported dimensions: {}".format(dimensions))

    return boundary_mask, missing_mask

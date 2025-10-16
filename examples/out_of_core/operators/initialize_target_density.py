from typing import Any
import warp as wp
import numpy as np
import stl as np_mesh


class InitializeTargetDensity:
    """
    Initialize target density operator.
    """

    def __init__(self, file_path: str, background_density: float, mesh_density: float):
        # Load the mesh
        mesh = np_mesh.Mesh.from_file(file_path)
        mesh_points = mesh.points.reshape(-1, 3)
        mesh_indices = np.arange(mesh_points.shape[0])
        self.mesh = wp.Mesh(
            points=wp.array(mesh_points, dtype=wp.vec3),
            indices=wp.array(mesh_indices, dtype=int),
        )
        self.background_density = background_density
        self.mesh_density = mesh_density

    @wp.kernel
    def _initialize_target_density(
        rho: wp.array4d(dtype=Any),
        mesh: wp.uint64,
        background_density: float,
        mesh_density: float,
        origin: wp.vec3f,
        spacing: wp.vec3f,
    ):
        # get spatial index
        i, j, k = wp.tid()

        # position of voxel (cell center)
        ijk = wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k))
        ijk = ijk + wp.vec3(0.5, 0.5, 0.5)  # cell center
        pos = wp.cw_mul(ijk, spacing) + origin

        # Compute maximum distance to check
        max_length = wp.sqrt(
            (spacing[0] * wp.float32(rho.shape[0])) ** 2.0
            + (spacing[1] * wp.float32(rho.shape[1])) ** 2.0
            + (spacing[2] * wp.float32(rho.shape[2])) ** 2.0
        )

        # evaluate distance of point
        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)
        p = wp.mesh_query_point_sign_winding_number(mesh, pos, max_length, sign, face_index, face_u, face_v)

        # set point to be solid
        if sign < 0.0:
            rho[0, i, j, k] = mesh_density
        else:
            rho[0, i, j, k] = background_density

    def __call__(
        self,
        rho,
        origin,
        spacing,
    ):
        # Voxelize STL of mesh
        wp.launch(
            self._initialize_target_density,
            inputs=[
                rho,
                wp.uint64(self.mesh.id),
                self.background_density,
                self.mesh_density,
                wp.vec3f(origin),
                wp.vec3f(spacing),
            ],
            dim=rho.shape[1:],
        )

        return rho

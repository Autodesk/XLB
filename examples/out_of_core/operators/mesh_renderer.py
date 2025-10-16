import warp as wp
import math


class MeshRenderer:
    """
    Operator for rendering a Warp mesh to a pixel and depth buffer using ray tracing.

    This operator takes a Warp mesh and renders it using ray tracing with Blinn-Phong
    shading, including diffuse, specular, and fresnel effects.

    Parameters
    ----------
    width : int
        Width of the output image in pixels
    height : int
        Height of the output image in pixels
    camera_position : wp.vec3
        Position of the camera in world space

    Attributes
    ----------
    Buffer Layout
    ------------
    pixel_buffer : wp.array3d(dtype=wp.float32)
        Shape: (height, width, 4)
        RGBA color buffer
    depth_buffer : wp.array2d(dtype=wp.float32)
        Shape: (height, width)
        Depth values in range [0,1]
    """

    @staticmethod
    @wp.func
    def create_view_matrix(eye: wp.vec3, target: wp.vec3, up: wp.vec3) -> wp.mat44:
        """Create a view matrix from camera parameters using right-handed coordinate system."""
        # Forward vector points from eye to target (negative z-axis in view space)
        forward = wp.normalize(target - eye)  # Note: reversed from before

        # Right vector
        right = wp.normalize(wp.cross(forward, up))

        # Recompute up vector to ensure orthogonality
        up = wp.normalize(wp.cross(right, forward))

        # Construct view matrix - note forward is negated to maintain right-handed system
        return wp.mat44(
            right[0], up[0], -forward[0], eye[0], right[1], up[1], -forward[1], eye[1], right[2], up[2], -forward[2], eye[2], 0.0, 0.0, 0.0, 1.0
        )

    @staticmethod
    @wp.func
    def normal_based_shading(
        normal: wp.vec3,
        view_dir: wp.vec3,
        base_color: wp.vec3,
        ambient_intensity: float,
        edge_sharpness: float,
    ) -> wp.vec3:
        """Compute lighting with simple normal-based shading."""
        # Normalize vectors
        n = wp.normalize(normal)
        v = wp.normalize(view_dir)

        # Check if normal is facing away from view direction
        n_dot_v = wp.dot(n, v)
        if n_dot_v < 0.0:
            # Flip normal if it's facing away
            n = wp.vec3(-n[0], -n[1], -n[2])
            n_dot_v = -n_dot_v

        # Use configurable falloff for edge definition
        diffuse_factor = wp.pow(n_dot_v, edge_sharpness)

        # Add ambient light to prevent completely black areas
        light = wp.vec3(ambient_intensity + diffuse_factor)

        # Apply lighting to base color using component-wise multiplication
        return wp.cw_mul(base_color, light)

    @wp.kernel
    def _render_mesh(
        mesh_id: wp.uint64,
        vertex_colors: wp.array2d(dtype=wp.float32),
        pixel_buffer: wp.array3d(dtype=wp.float32),
        depth_buffer: wp.array2d(dtype=wp.float32),
        camera_pos: wp.vec3f,
        camera_target: wp.vec3f,
        camera_up: wp.vec3f,
        fov_degrees: float,
        ambient_intensity: float,
        edge_sharpness: float,
        gamma: float,
    ):
        # Get pixel coordinates
        i, j = wp.tid()
        height = pixel_buffer.shape[0]
        width = pixel_buffer.shape[1]

        # Get mesh
        mesh = wp.mesh_get(mesh_id)

        # Convert FOV to radians and calculate image plane parameters
        aspect = float(width) / float(height)
        fov = math.radians(fov_degrees)
        tan_fov = math.tan(fov * 0.5)

        # Convert to NDC space with proper FOV
        sx = (2.0 * float(j) / float(width) - 1.0) * aspect * tan_fov
        sy = (1.0 - 2.0 * float(i) / float(height)) * tan_fov

        # Create view matrix
        view = MeshRenderer.create_view_matrix(camera_pos, camera_target, camera_up)

        # Create ray in camera space
        ray_dir = wp.normalize(wp.vec3(sx, sy, -1.0))

        # Transform ray to world space
        ro = camera_pos
        rd = wp.vec3(
            ray_dir[0] * view[0, 0] + ray_dir[1] * view[0, 1] + ray_dir[2] * view[0, 2],
            ray_dir[0] * view[1, 0] + ray_dir[1] * view[1, 1] + ray_dir[2] * view[1, 2],
            ray_dir[0] * view[2, 0] + ray_dir[1] * view[2, 1] + ray_dir[2] * view[2, 2],
        )
        rd = wp.normalize(rd)

        # Ray trace against mesh
        query = wp.mesh_query_ray(mesh_id, ro, rd, depth_buffer[i, j])
        if query.result:
            if query.t < depth_buffer[i, j]:
                # Use normal-based coloring for debugging
                normal = wp.normalize(query.normal)

                # Get indices of the face
                i0 = mesh.indices[3 * query.face + 0]  # First vertex
                i1 = mesh.indices[3 * query.face + 1]  # Second vertex
                i2 = mesh.indices[3 * query.face + 2]  # Third vertex

                # Get vertex colors
                c0 = wp.vec3(vertex_colors[i0, 0], vertex_colors[i0, 1], vertex_colors[i0, 2])
                c1 = wp.vec3(vertex_colors[i1, 0], vertex_colors[i1, 1], vertex_colors[i1, 2])
                c2 = wp.vec3(vertex_colors[i2, 0], vertex_colors[i2, 1], vertex_colors[i2, 2])

                # Use barycentric coordinates from query
                w0 = query.u  # Weight for first edge (between v0 and v1)
                w1 = query.v  # Weight for second edge (between v1 and v2)
                w2 = 1.0 - query.u - query.v  # Weight for remaining vertex

                # Interpolate vertex colors using barycentric coordinates
                base_color = (
                    wp.cw_mul(c0, wp.vec3(w0))  # First vertex
                    + wp.cw_mul(c1, wp.vec3(w1))  # Second vertex
                    + wp.cw_mul(c2, wp.vec3(w2))  # Third vertex
                )

                # Compute lighting
                color = MeshRenderer.normal_based_shading(
                    normal=normal,
                    view_dir=rd,
                    base_color=base_color,
                    ambient_intensity=ambient_intensity,
                    edge_sharpness=edge_sharpness,
                )

                # Apply gamma correction (linear to sRGB)
                color = wp.vec3(
                    wp.pow(wp.clamp(color[0], 0.0, 1.0), 1.0 / gamma),
                    wp.pow(wp.clamp(color[1], 0.0, 1.0), 1.0 / gamma),
                    wp.pow(wp.clamp(color[2], 0.0, 1.0), 1.0 / gamma),
                )

                # Write results
                pixel_buffer[i, j, 0] = color[0]
                pixel_buffer[i, j, 1] = color[1]
                pixel_buffer[i, j, 2] = color[2]
                pixel_buffer[i, j, 3] = 1.0
                depth_buffer[i, j] = query.t

    def __call__(
        self,
        mesh: wp.Mesh,
        vertex_colors: wp.array2d,  # Shape: (num_vertices, 3) for RGB colors
        pixel_buffer: wp.array3d,
        depth_buffer: wp.array2d,
        camera_pos: wp.vec3f = wp.vec3f(0.0, 1.0, 2.0),
        camera_target: wp.vec3f = wp.vec3f(0.0, 0.0, 0.0),
        camera_up: wp.vec3f = wp.vec3f(0.0, 1.0, 0.0),
        fov_degrees: float = 60.0,
        ambient_intensity: float = 0.05,
        edge_sharpness: float = 1.0,
        gamma: float = 1.0,
    ):
        """
        Render a Warp mesh with normal-based shading.

        Parameters
        ----------
        mesh : wp.Mesh
            Warp mesh to render
        vertex_colors : wp.array2d
            Vertex colors (num_vertices, 3) for RGB colors
        pixel_buffer : wp.array3d
            Output pixel buffer (height, width, 4) RGBA
        depth_buffer : wp.array2d
            Output depth buffer (height, width)
        camera_pos : wp.vec3f
            Camera position in world space
        camera_target : wp.vec3f
            Point the camera is looking at
        camera_up : wp.vec3f
            Camera up vector
        fov_degrees : float
            Field of view in degrees
        ambient_intensity : float
            Intensity of ambient light (0.0-1.0)
        edge_sharpness : float
            Controls edge definition (lower values = softer edges, higher values = sharper edges)
        gamma : float
            Gamma correction value (typically 1.0-2.2, lower = brighter)

        Returns
        -------
        tuple[wp.array3d, wp.array2d]
            Updated pixel and depth buffers
        """
        # Launch kernel
        wp.launch(
            self._render_mesh,
            dim=(pixel_buffer.shape[0], pixel_buffer.shape[1]),
            inputs=[
                mesh.id,
                vertex_colors,
                pixel_buffer,
                depth_buffer,
                camera_pos,
                camera_target,
                camera_up,
                fov_degrees,
                ambient_intensity,
                edge_sharpness,
                gamma,
            ],
        )

        return pixel_buffer, depth_buffer

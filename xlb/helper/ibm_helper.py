from xlb import DefaultConfig
from xlb.grid import grid_factory
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import Precision
from typing import Tuple
import warp as wp
import trimesh
import numpy as np


def create_ibm_fields(grid_shape: Tuple[int, int, int], velocity_set=None, precision_policy=None):
    velocity_set = velocity_set or DefaultConfig.velocity_set
    compute_backend = ComputeBackend.WARP
    precision_policy = precision_policy or DefaultConfig.default_precision_policy
    grid = grid_factory(grid_shape, compute_backend=compute_backend)

    # Create fields
    f_0 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    f_1 = grid.create_field(cardinality=velocity_set.q, dtype=precision_policy.store_precision)
    velocity_eulerian = grid.create_field(cardinality=3, dtype=precision_policy.store_precision)
    missing_mask = grid.create_field(cardinality=velocity_set.q, dtype=Precision.BOOL)
    bc_mask = grid.create_field(cardinality=1, dtype=Precision.UINT8)

    return grid, f_0, f_1, missing_mask, bc_mask


def transform_mesh(mesh, translation=None, rotation=None, rotation_order="xyz", scale=None):
    """
    Transform a mesh using translation, rotation, and scaling.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input triangle mesh
    translation : array-like or None, shape (3,)
        Translation vector [x, y, z]. If None, no translation is applied
    rotation : array-like or None, shape (3,)
        Rotation angles in degrees [rx, ry, rz]. If None, no rotation is applied
    rotation_order : str, default='xyz'
        Order of rotations. Valid options: 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'
    scale : float or array-like or None, shape (3,)
        Scale factor. If float, uniform scaling is applied.
        If array-like, [sx, sy, sz] for non-uniform scaling.
        If None, no scaling is applied

    Returns
    -------
    trimesh.Trimesh
        Transformed mesh
    """
    # Create a copy of the mesh to avoid modifying the original
    transformed_mesh = mesh.copy()

    # Apply scaling
    if scale is not None:
        if isinstance(scale, (int, float)):
            scale = [scale, scale, scale]
        transformed_mesh.apply_scale(scale)

    # Apply rotation
    if rotation is not None:
        # Convert degrees to radians
        rotation = np.array(rotation) * np.pi / 180.0

        # Create rotation matrix based on the specified order
        matrix = trimesh.transformations.euler_matrix(rotation[0], rotation[1], rotation[2], axes=f"r{rotation_order}")
        transformed_mesh.apply_transform(matrix)

    # Apply translation
    if translation is not None:
        translation_matrix = trimesh.transformations.translation_matrix(translation)
        transformed_mesh.apply_transform(translation_matrix)

    return transformed_mesh


def prepare_immersed_boundary(mesh, max_lbm_length, translation=None, rotation=None, rotation_order="xyz", scale=None):
    """
    Prepare an immersed boundary from an STL file with optional transformations.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input triangle mesh
    max_lbm_length : float
        Desired maximum length in lattice units
    translation : array-like or None, shape (3,)
        Translation vector [x, y, z]. If None, no translation is applied
    rotation : array-like or None, shape (3,)
        Rotation angles in degrees [rx, ry, rz]. If None, no rotation is applied
    rotation_order : str, default='xyz'
        Order of rotations. Valid options: 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'
    scale : float or array-like or None, shape (3,)
        Additional scale factor applied after normalization.
        If float, uniform scaling is applied.
        If array-like, [sx, sy, sz] for non-uniform scaling.
        If None, no additional scaling is applied

    Returns
    -------
    tuple
        (vertices_wp, vertex_areas_wp, faces_np)
        - vertices_wp: Warp array containing vertex coordinates
        - vertex_areas_wp: Warp array containing Voronoi areas for each vertex
        - faces_np: NumPy array containing face indices
    """

    # Subdivide to ensure at least one vertex per cell
    mesh = mesh.subdivide_to_size(max_edge=1.0, max_iter=200)

    # Calculate vertices and voronoi areas
    vertices_wp, vertex_areas_wp = calculate_voronoi_areas(mesh)

    # Return the faces along with vertices and areas
    return vertices_wp, vertex_areas_wp, mesh.faces


def calculate_voronoi_areas(mesh, check_area=True):
    """
    Calculate Voronoi areas for vertices in a triangle mesh using Warp.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input triangle mesh
    check_area : bool, optional
        Whether to check if the sum of the Voronoi areas matches the mesh area

    Returns
    -------
    tuple
        (vertex_areas_wp, vertex_areas)
        - vertex_areas_wp: Warp array containing Voronoi areas for each vertex
        - vertex_areas: NumPy array containing Voronoi areas for each vertex
    """
    # Get face areas and vertices of each face
    face_areas = mesh.area_faces
    faces = mesh.faces
    vertices = mesh.vertices

    # Define the number of vertices and faces
    num_vertices = len(vertices)
    num_faces = len(faces)

    @wp.kernel
    def voronoi_area_kernel(
        faces: wp.array2d(dtype=int), vertices: wp.array(dtype=wp.vec3), face_areas: wp.array1d(dtype=float), vertex_areas: wp.array1d(dtype=float)
    ):
        tid = wp.tid()

        # Get vertex indices of the face
        v0 = faces[tid, 0]
        v1 = faces[tid, 1]
        v2 = faces[tid, 2]

        # Get vertex positions
        p0 = wp.vec3(vertices[v0][0], vertices[v0][1], vertices[v0][2])
        p1 = wp.vec3(vertices[v1][0], vertices[v1][1], vertices[v1][2])
        p2 = wp.vec3(vertices[v2][0], vertices[v2][1], vertices[v2][2])

        # Compute edge lengths of the triangle
        a = wp.length(p1 - p2)
        b = wp.length(p0 - p2)
        c = wp.length(p0 - p1)

        # Compute area and cotangent weights
        face_area = face_areas[tid]

        cot_alpha = (b**2.0 + c**2.0 - a**2.0) / (4.0 * face_area)
        cot_beta = (a**2.0 + c**2.0 - b**2.0) / (4.0 * face_area)
        cot_gamma = (a**2.0 + b**2.0 - c**2.0) / (4.0 * face_area)

        # Normalize the cotangent weights
        total_cot = cot_alpha + cot_beta + cot_gamma
        if total_cot > 0:
            cot_alpha /= total_cot
            cot_beta /= total_cot
            cot_gamma /= total_cot

        # Distribute the face area to each vertex based on the normalized weights
        wp.atomic_add(vertex_areas, v0, face_area * cot_beta / 2.0 + face_area * cot_gamma / 2.0)
        wp.atomic_add(vertex_areas, v1, face_area * cot_alpha / 2.0 + face_area * cot_gamma / 2.0)
        wp.atomic_add(vertex_areas, v2, face_area * cot_alpha / 2.0 + face_area * cot_beta / 2.0)

    # Convert data to Warp arrays
    faces_wp = wp.array(faces, dtype=wp.int32)
    vertices_wp = wp.array(vertices, dtype=wp.vec3)
    face_areas_wp = wp.array(face_areas, dtype=wp.float32)
    vertex_areas_wp = wp.zeros(num_vertices, dtype=wp.float32)

    # Launch the kernel
    wp.launch(kernel=voronoi_area_kernel, dim=num_faces, inputs=[faces_wp, vertices_wp, face_areas_wp, vertex_areas_wp], device="cuda")

    # Validate the result
    if check_area:
        vertex_areas_np = vertex_areas_wp.numpy()
        if abs(vertex_areas_np.sum() - mesh.area) > 1e-2:
            # Copy the result back to the CPU
            print("Warning: Sum of Voronoi areas does not match mesh area")
        else:
            print("Voronoi areas calculated successfully")
        print(f"Sum of Voronoi areas: {vertex_areas_np.sum()}")
        print(f"Mesh area: {mesh.area}")

    return vertices_wp, vertex_areas_wp


def reconstruct_mesh_from_vertices_and_faces(vertices_wp, faces_np, save_path=None):
    """
    Reconstruct a trimesh from Warp vertices and NumPy-based faces.

    Parameters
    ----------
    vertices_wp : wp.array
        Warp array containing vertex coordinates
    faces_np : np.ndarray
        NumPy array containing face indices (each row is a triangle with 3 vertex indices)
    save_path : str or None, optional
        If provided, saves the mesh to this path with .stl extension

    Returns
    -------
    trimesh.Trimesh
        Reconstructed mesh
    """
    # Convert Warp vertices to numpy
    vertices = vertices_wp.numpy()

    # Create the mesh using the provided faces
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces_np)

    # Save if path is provided
    if save_path is not None:
        if not save_path.endswith(".stl"):
            save_path += ".stl"
        mesh.export(save_path)
        print(f"Mesh saved to: {save_path}")

    return mesh

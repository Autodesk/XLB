import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from time import time
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from jax.image import resize
from jax import jit
import jax.numpy as jnp
from functools import partial
import trimesh
import vtk
import open3d as o3d
import h5py

import os
import __main__


@partial(jit, static_argnums=(1, 2))
def downsample_field(field, factor, method="bicubic"):
    """
    Downsample a JAX array by a factor of `factor` along each axis.

    Parameters
    ----------
    field : jax.numpy.ndarray
        The input vector field to be downsampled. This should be a 3D or 4D JAX array where the last dimension is 2 or 3 (vector components).
    factor : int
        The factor by which to downsample the field. The dimensions of the field will be divided by this factor.
    method : str, optional
        The method to use for downsampling. Default is 'bicubic'.

    Returns
    -------
    jax.numpy.ndarray
        The downsampled field.
    """
    if factor == 1:
        return field
    else:
        new_shape = tuple(dim // factor for dim in field.shape[:-1])
        downsampled_components = []
        for i in range(field.shape[-1]):  # Iterate over the last dimension (vector components)
            resized = resize(field[..., i], new_shape, method=method)
            downsampled_components.append(resized)

        return jnp.stack(downsampled_components, axis=-1)


def save_image(fld, timestep=None, prefix=None, **kwargs):
    """
    Save an image of a field at a given timestep.

    Parameters
    ----------
    timestep : int
        The timestep at which the field is being saved.
    fld : jax.numpy.ndarray
        The field to be saved. This should be a 2D or 3D JAX array. If the field is 3D, the magnitude of the field will be calculated and saved.
    prefix : str, optional
        A prefix to be added to the filename. The filename will be the name of the main script file by default.

    Returns
    -------
    None

    Notes
    -----
    This function saves the field as an image in the PNG format.
    The filename is based on the name of the main script file, the provided prefix, and the timestep number.
    If the field is 3D, the magnitude of the field is calculated and saved.
    The image is saved with the 'nipy_spectral' colormap and the origin set to 'lower'.
    """
    if prefix is None:
        fname = os.path.basename(__main__.__file__)
        fname = os.path.splitext(fname)[0]
    else:
        fname = prefix

    if timestep is not None:
        fname = fname + "_" + str(timestep).zfill(4)

    if len(fld.shape) > 3:
        raise ValueError("The input field should be 2D!")
    if len(fld.shape) == 3:
        fld = np.sqrt(fld[0, ...] ** 2 + fld[0, ...] ** 2)

    plt.clf()
    kwargs.pop("cmap", None)
    plt.imsave(fname + ".png", fld.T, cmap=cm.nipy_spectral, origin="lower", **kwargs)


def save_fields_vtk(fields, timestep, output_dir=".", prefix="fields", shift_coords=(0, 0, 0), scale=1):
    """
    Save VTK fields to the specified directory, shifting the coordinates if needed.

    Parameters
    ----------
    timestep (int): The timestep number to be associated with the saved fields.
    fields (Dict[str, np.ndarray]): A dictionary of fields to be saved. Each field must be an array-like object
        with dimensions (nx, ny) for 2D fields or (nx, ny, nz) for 3D fields, where:
            - nx : int, number of grid points along the x-axis
            - ny : int, number of grid points along the y-axis
            - nz : int, number of grid points along the z-axis (for 3D fields only)
        The key value for each field in the dictionary must be a string containing the name of the field.
    output_dir (str, optional, default: '.'): The directory in which to save the VTK files. Defaults to the current directory.
    prefix (str, optional, default: 'fields'): A prefix to be added to the filename. Defaults to 'fields'.
    shift_coords (tuple, optional, default: (0, 0, 0)): The amount to shift in the x, y, and z directions.
    scale (int, optional, default: 1): The amount to scale the geometry.

    Returns
    -------
    None

    Notes
    -----
    This function saves the VTK fields in the specified directory, with filenames based on the provided timestep number
    and the filename. For example, if the timestep number is 10 and the file name is fields, the VTK file
    will be saved as 'fields_0000010.vtk'in the specified directory.

    """
    start = time()
    # Assert that all fields have the same dimensions
    for key, value in fields.items():
        if key == list(fields.keys())[0]:
            dimensions = value.shape
        else:
            assert value.shape == dimensions, "All fields must have the same dimensions!"

    output_filename = os.path.join(output_dir, prefix + "_" + f"{timestep:08d}.vtk")

    # Add 1 to the dimensions tuple as we store cell values
    dimensions = tuple([dim + 1 for dim in dimensions])

    # Create a uniform grid
    if value.ndim == 2:
        dimensions = dimensions + (1,)

    grid = pv.ImageData(dimensions=dimensions, origin=shift_coords, spacing=(scale, scale, scale))

    # Add the fields to the grid
    for key, value in fields.items():
        grid[key] = value.flatten(order="F")

    # Save the grid to a VTK file
    grid.save(output_filename, binary=True)
    print(f"Saved {output_filename} in {time() - start:.6f} seconds.")


def map_field_vtk_interpolator(
    field, stl_filename, voxel_size, output_dir=".", prefix="mapped_field", origin=[0, 0, 0], method="cubic", normals=True
):
    """
    Map a volumetric field onto an STL mesh using RegularGridInterpolator.

    Parameters
    ----------
    field : np.ndarray
        3D array representing the volumetric field.
    stl_filename : str
        Path to the STL file.
    voxel_size : float
        Size of a voxel along each axis.
    output_dir : str, optional
        Directory to save the output VTK file.
    prefix : str, optional
        Filename prefix.
    origin : list or tuple of float, optional
        Origin of the grid.
    method : str, optional
        Interpolation method (e.g., 'cubic').
    normals : bool, optional
        If True, use normal-direction averaging by sampling points offset along the surface normal;
        if False, simply sample the field at the surface points.

    Returns
    -------
    None
    """

    print("Mapping field to stl with {} method".format("normal averaging" if normals else "original sampling"))
    start = time()
    grid_shape = field.shape

    # Create coordinate arrays based on the origin and voxel size.
    x = origin[0] + np.arange(grid_shape[0]) * voxel_size
    y = origin[1] + np.arange(grid_shape[1]) * voxel_size
    z = origin[2] + np.arange(grid_shape[2]) * voxel_size

    # Set up the interpolation function.
    interp_func = RegularGridInterpolator((x, y, z), field, method=method, bounds_error=False, fill_value=None)

    # Load the STL mesh.
    stl_mesh = pv.read(stl_filename)

    if normals:
        # Compute normals if not already available.
        if "Normals" not in stl_mesh.point_data:
            stl_mesh = stl_mesh.compute_normals()
        normals_arr = stl_mesh.point_normals  # shape (N, 3)
        points = stl_mesh.points  # shape (N, 3)

        # Define offsets along the normal: sample 2 voxels in both directions including the surface.
        offsets = np.array([-2, 2]) * voxel_size  # shape (5,)
        # offsets = np.array([-2, -1, 0, 1, 2]) * voxel_size  # shape (5,)
        # Generate sample points along the normal for each mesh point.
        sample_points = points[:, np.newaxis, :] + offsets[np.newaxis, :, np.newaxis] * normals_arr[:, np.newaxis, :]
        sample_points_reshaped = sample_points.reshape(-1, 3)

        # Interpolate the field at each of the sample points.
        field_values = interp_func(sample_points_reshaped)
        field_values = field_values.reshape(points.shape[0], len(offsets))
        # Average the values along the normal offset direction.
        field_mapped = np.mean(field_values, axis=1)
    else:
        # Original: simply sample the field at the surface points.
        points = stl_mesh.points
        field_mapped = interp_func(points)

    # Assign the mapped field to the mesh and save.
    stl_mesh["field"] = field_mapped
    output_filename = os.path.join(output_dir, prefix + ".vtk")
    stl_mesh.save(output_filename)
    print(f"Saved {output_filename} in {time() - start:.6f} seconds.")


def map_field_vtk(field, stl_filename, output_dir=".", prefix="mapped_field", shift_coords=(0, 0, 0), scale=1, normals=True):
    """
    Save VTK fields to the specified directory by probing a uniform grid
    generated from a field array onto an STL mesh. If normals is True, for
    each STL point the field is averaged over points offset along the surface normal.

    Parameters
    ----------
    field : np.ndarray
        The field data (2D or 3D) to be mapped.
    stl_filename : str
        Path to the STL file.
    output_dir : str, optional
        Directory to save the output VTK file.
    prefix : str, optional
        Filename prefix.
    shift_coords : tuple, optional
        Origin (shift) for the uniform grid.
    scale : int or float, optional
        Spacing of the uniform grid.
    normals : bool, optional
        If True, average field values along the surface normal (sampling 2 voxels on either side);
        if False, use the original probe method.

    Returns
    -------
    None
    """
    start = time()
    method_str = "normal averaging" if normals else "original sampling"
    print(f"Mapping field to stl with {method_str}")
    output_filename = os.path.join(output_dir, prefix + ".vtk")

    # Create the uniform grid dimensions (note: cell values require dimensions + 1).
    dimensions = tuple(dim + 1 for dim in field.shape)
    if field.ndim == 2:
        dimensions = dimensions + (1,)

    # Create a uniform grid (ImageData) with the specified origin and spacing.
    grid = pv.ImageData(dimensions=dimensions, origin=shift_coords, spacing=(scale, scale, scale))
    grid.cell_data["field"] = field.flatten(order="F")
    grid = grid.cell_data_to_point_data()

    # Load the STL mesh.
    stl_mesh = pv.read(stl_filename)

    if normals:
        # Compute normals if not available.
        if "Normals" not in stl_mesh.point_data:
            stl_mesh = stl_mesh.compute_normals()
        normals_arr = stl_mesh.point_normals  # shape (N, 3)
        points = stl_mesh.points  # shape (N, 3)

        # Define offsets along the normal: sample 2 voxels in both directions.
        offsets = np.array([-2, 2]) * scale
        # offsets = np.array([-2, -1, 0, 1, 2]) * scale
        # Generate sample points along the normal for each STL point.
        sample_points = points[:, np.newaxis, :] + offsets[np.newaxis, :, np.newaxis] * normals_arr[:, np.newaxis, :]
        sample_points_reshaped = sample_points.reshape(-1, 3)

        # Create a PolyData object from these sample points.
        samples_pd = pv.PolyData(sample_points_reshaped)

        # Use vtkProbeFilter to sample the grid at these sample locations.
        probe = vtk.vtkProbeFilter()
        probe.SetInputData(samples_pd)
        probe.SetSourceData(grid)
        probe.Update()
        sampled = pv.wrap(probe.GetOutput())
        sample_field = sampled.point_data["field"]
        averaged_field = np.mean(sample_field.reshape(-1, len(offsets)), axis=1)

        # Assign the averaged field to the mesh.
        stl_mesh["field"] = averaged_field
        stl_mesh.save(output_filename)
    else:
        # Original method: use vtkProbeFilter on the STL geometry.
        stl_vtk = stl_mesh.extract_geometry()
        probe = vtk.vtkProbeFilter()
        probe.SetInputData(stl_vtk)
        probe.SetSourceData(grid)
        probe.Update()
        stl_mapped = pv.wrap(probe.GetOutput())
        stl_mapped.save(output_filename)

    print(f"Saved {output_filename} in {time() - start:.6f} seconds.")


def save_BCs_vtk(timestep, BCs, gridInfo, output_dir="."):
    """
    Save boundary conditions as VTK format to the specified directory.

    Parameters
    ----------
    timestep (int): The timestep number to be associated with the saved fields.
    BCs (List[BC]): A list of boundary conditions to be saved. Each boundary condition must be an object of type BC.

    Returns
    -------
    None

    Notes
    -----
    This function saves the boundary conditions in the specified directory, with filenames based on the provided timestep number
    and the filename. For example, if the timestep number is 10, the VTK file
    will be saved as 'BCs_0000010.vtk'in the specified directory.
    """

    # Create a uniform grid
    if gridInfo["nz"] == 0:
        gridDimensions = (gridInfo["nx"] + 1, gridInfo["ny"] + 1, 1)
        fieldDimensions = (gridInfo["nx"], gridInfo["ny"], 1)
    else:
        gridDimensions = (gridInfo["nx"] + 1, gridInfo["ny"] + 1, gridInfo["nz"] + 1)
        fieldDimensions = (gridInfo["nx"], gridInfo["ny"], gridInfo["nz"])

    grid = pv.ImageData(dimensions=gridDimensions)

    # Dictionary to keep track of encountered BC names
    bcNamesCount = {}

    for bc in BCs:
        bcName = bc.name
        if bcName in bcNamesCount:
            bcNamesCount[bcName] += 1
        else:
            bcNamesCount[bcName] = 0
        bcName += f"_{bcNamesCount[bcName]}"

        if bc.isDynamic:
            bcIndices, _ = bc.update_function(timestep)
        else:
            bcIndices = bc.indices

        # Convert indices to 1D indices
        if gridInfo["dim"] == 2:
            bcIndices = np.ravel_multi_index(bcIndices, fieldDimensions[:-1], order="F")
        else:
            bcIndices = np.ravel_multi_index(bcIndices, fieldDimensions, order="F")

        grid[bcName] = np.zeros(fieldDimensions, dtype=bool).flatten(order="F")
        grid[bcName][bcIndices] = True

    # Save the grid to a VTK file
    output_filename = os.path.join(output_dir, "BCs_" + f"{timestep:07d}.vtk")

    start = time()
    grid.save(output_filename, binary=True)
    print(f"Saved {output_filename} in {time() - start:.6f} seconds.")


def rotate_geometry(indices, origin, axis, angle):
    """
    Rotates a voxelized mesh around a given axis.

    Parameters
    ----------
    indices : array-like
        The indices of the voxels in the mesh.
    origin : array-like
        The coordinates of the origin of the rotation axis.
    axis : array-like
        The direction vector of the rotation axis. This should be a 3-element sequence.
    angle : float
        The angle by which to rotate the mesh, in radians.

    Returns
    -------
    tuple
        The indices of the voxels in the rotated mesh.

    Notes
    -----
    This function rotates the mesh by applying a rotation matrix to the voxel indices. The rotation matrix is calculated
    using the axis-angle representation of rotations. The origin of the rotation axis is assumed to be at (0, 0, 0).
    """
    indices_rotated = (jnp.array(indices).T - origin) @ axangle2mat(axis, angle) + origin
    return tuple(jnp.rint(indices_rotated).astype("int32").T)


def voxelize_stl(stl_filename, length_lbm_unit=None, tranformation_matrix=None, pitch=None):
    """
    Converts an STL file to a voxelized mesh.

    Parameters
    ----------
    stl_filename : str
        The name of the STL file to be voxelized.
    length_lbm_unit : float, optional
        The unit length in LBM. Either this or 'pitch' must be provided.
    tranformation_matrix : array-like, optional
        A transformation matrix to be applied to the mesh before voxelization.
    pitch : float, optional
        The pitch of the voxel grid. Either this or 'length_lbm_unit' must be provided.

    Returns
    -------
    trimesh.VoxelGrid, float
        The voxelized mesh and the pitch of the voxel grid.

    Notes
    -----
    This function uses the trimesh library to load the STL file and voxelized the mesh. If a transformation matrix is
    provided, it is applied to the mesh before voxelization. The pitch of the voxel grid is calculated based on the
    maximum extent of the mesh and the provided lattice Boltzmann unit length, unless a pitch is provided directly.
    """
    if length_lbm_unit is None and pitch is None:
        raise ValueError("Either 'length_lbm_unit' or 'pitch' must be provided!")
    mesh = trimesh.load_mesh(stl_filename, process=False)
    length_phys_unit = mesh.extents.max()
    if tranformation_matrix is not None:
        mesh.apply_transform(tranformation_matrix)
    if pitch is None:
        pitch = length_phys_unit / length_lbm_unit
    mesh_voxelized = mesh.voxelized(pitch=pitch)
    return mesh_voxelized, pitch


def save_fields_hdf5(fields, timestep, output_dir=".", prefix="fields", shift_coords=(0, 0, 0), scale=1, compression="gzip", compression_opts=0):
    start = time()
    filename = str(prefix + "_" + f"{timestep:08d}.h5")
    output_filename = os.path.join(output_dir, filename)

    # Determine the dimensions (assuming all fields have the same shape)
    for key, value in fields.items():
        if key == list(fields.keys())[0]:
            dimensions = value.shape
        else:
            assert value.shape == dimensions, "All fields must have the same dimensions!"

    with h5py.File(output_filename, "w") as f:
        # Write field data with Fortran order to match the VTK convention
        for key, value in fields.items():
            value = np.transpose(value, (2, 1, 0))

            dataset = f.create_dataset(key, data=value, dtype="float32", compression=compression, compression_opts=compression_opts)
            dataset.attrs["origin"] = shift_coords
            dataset.attrs["spacing"] = (scale, scale, scale)

    # Write the XDMF file using HyperSlab to properly reference the HDF5 data
    xdmf_filename = os.path.join(output_dir, prefix + "_" + f"{timestep:08d}.xdmf")
    with open(xdmf_filename, "w") as xdmf:
        xdmf.write(f"""<?xml version="1.0" ?>
<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="fields" GridType="Uniform">
      <Topology TopologyType="3DCoRectMesh" Dimensions="{dimensions[2] + 1} {dimensions[1] + 1} {dimensions[0] + 1}"/>
      <Geometry GeometryType="ORIGIN_DXDYDZ">
        <DataItem Dimensions="3" NumberType="Float" Precision="4" Format="XML">
          {shift_coords[2]} {shift_coords[1]} {shift_coords[0]}
        </DataItem>
        <DataItem Dimensions="3" NumberType="Float" Precision="4" Format="XML">
          {scale} {scale} {scale}
        </DataItem>
      </Geometry>
""")
        for key in fields.keys():
            xdmf.write(f"""
      <Attribute Name="{key}" AttributeType="Scalar" Center="Cell">
        <DataItem ItemType="HyperSlab" Dimensions="{dimensions[2]} {dimensions[1]} {dimensions[0]}" NumberType="Float" Precision="4" Format="HDF">
          <DataItem Dimensions="3 3" Format="XML">
            0 0 0
            1 1 1
            {dimensions[2]} {dimensions[1]} {dimensions[0]}
          </DataItem>
          <DataItem Dimensions="{dimensions[2]} {dimensions[1]} {dimensions[0]}" NumberType="Float" Precision="4" Format="HDF">
            {filename}:/{key}
          </DataItem>
        </DataItem>
      </Attribute>
""")
        xdmf.write("""
    </Grid>
  </Domain>
</Xdmf>
""")

    print(f"Saved {output_filename} and {xdmf_filename} in {time() - start:.6f} seconds.")


def axangle2mat(axis, angle, is_normalized=False):
    """Rotation matrix for rotation angle `angle` around `axis`
    Parameters
    ----------
    axis : 3 element sequence
       vector specifying axis for rotation.
    angle : scalar
       angle of rotation in radians.
    is_normalized : bool, optional
       True if `axis` is already normalized (has norm of 1).  Default False.
    Returns
    -------
    mat : array shape (3,3)
       rotation matrix for specified rotation
    Notes
    -----
    From : https://github.com/matthew-brett/transforms3d
    Ref : http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    x, y, z = axis
    if not is_normalized:
        n = jnp.sqrt(x * x + y * y + z * z)
        x = x / n
        y = y / n
        z = z / n
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    C = 1 - c
    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC
    return jnp.array([
        [x * xC + c, xyC - zs, zxC + ys],
        [xyC + zs, y * yC + c, yzC - xs],
        [zxC - ys, yzC + xs, z * zC + c],
    ])


def voxelize_stl_open3d(stl_filename, length_lbm_unit):
    # Load the STL file
    mesh = o3d.io.read_triangle_mesh(stl_filename)
    print("..Model read")
    # Compute the voxel grid from the mesh
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=length_lbm_unit)
    print("...Grid created")
    # Get the bounding box of the voxel grid
    bbox = voxel_grid.get_axis_aligned_bounding_box()

    # Calculate the number of voxels along each axis
    grid_size = np.ceil((bbox.get_max_bound() - bbox.get_min_bound()) / length_lbm_unit).astype(int)

    # Initialize an empty 3D array based on the calculated grid size
    voxel_matrix = np.zeros(grid_size, dtype=bool)

    # Convert voxel indices to a boolean matrix
    for voxel in voxel_grid.get_voxels():
        x, y, z = voxel.grid_index
        voxel_matrix[x, y, z] = True

    # Return the voxel matrix and the bounding box corners
    return voxel_matrix, bbox.get_box_points()


@partial(jit)
def q_criterion(u, omega=2.0):
    # Compute derivatives
    u_x = u[0, ...]
    u_y = u[1, ...]
    u_z = u[2, ...]

    # Compute derivatives
    u_x_dx = (u_x[2:, 1:-1, 1:-1] - u_x[:-2, 1:-1, 1:-1]) / 2
    u_x_dy = (u_x[1:-1, 2:, 1:-1] - u_x[1:-1, :-2, 1:-1]) / 2
    u_x_dz = (u_x[1:-1, 1:-1, 2:] - u_x[1:-1, 1:-1, :-2]) / 2
    u_y_dx = (u_y[2:, 1:-1, 1:-1] - u_y[:-2, 1:-1, 1:-1]) / 2
    u_y_dy = (u_y[1:-1, 2:, 1:-1] - u_y[1:-1, :-2, 1:-1]) / 2
    u_y_dz = (u_y[1:-1, 1:-1, 2:] - u_y[1:-1, 1:-1, :-2]) / 2
    u_z_dx = (u_z[2:, 1:-1, 1:-1] - u_z[:-2, 1:-1, 1:-1]) / 2
    u_z_dy = (u_z[1:-1, 2:, 1:-1] - u_z[1:-1, :-2, 1:-1]) / 2
    u_z_dz = (u_z[1:-1, 1:-1, 2:] - u_z[1:-1, 1:-1, :-2]) / 2

    # Compute vorticity
    mu_x = u_z_dy - u_y_dz
    mu_y = u_x_dz - u_z_dx
    mu_z = u_y_dx - u_x_dy
    norm_mu = jnp.sqrt(mu_x**2 + mu_y**2 + mu_z**2)

    # Compute strain rate
    s_0_0 = u_x_dx
    s_0_1 = 0.5 * (u_x_dy + u_y_dx)
    s_0_2 = 0.5 * (u_x_dz + u_z_dx)
    s_1_0 = s_0_1
    s_1_1 = u_y_dy
    s_1_2 = 0.5 * (u_y_dz + u_z_dy)
    s_2_0 = s_0_2
    s_2_1 = s_1_2
    s_2_2 = u_z_dz
    s_dot_s = s_0_0**2 + s_0_1**2 + s_0_2**2 + s_1_0**2 + s_1_1**2 + s_1_2**2 + s_2_0**2 + s_2_1**2 + s_2_2**2

    # Compute Viscosity from Omega
    mu = ((1 / omega) - 0.5) / 3.0

    # Compute shear stress components
    tau_xy = 2 * mu * s_0_1
    tau_xz = 2 * mu * s_0_2
    tau_yz = 2 * mu * s_1_2

    # Compute shear stress magnitude
    tau_magnitude = jnp.sqrt(tau_xy**2 + tau_xz**2 + tau_yz**2)

    # Compute omega
    omega_0_0 = 0.0
    omega_0_1 = 0.5 * (u_x_dy - u_y_dx)
    omega_0_2 = 0.5 * (u_x_dz - u_z_dx)
    omega_1_0 = -omega_0_1
    omega_1_1 = 0.0
    omega_1_2 = 0.5 * (u_y_dz - u_z_dy)
    omega_2_0 = -omega_0_2
    omega_2_1 = -omega_1_2
    omega_2_2 = 0.0
    omega_dot_omega = (
        omega_0_0**2 + omega_0_1**2 + omega_0_2**2 + omega_1_0**2 + omega_1_1**2 + omega_1_2**2 + omega_2_0**2 + omega_2_1**2 + omega_2_2**2
    )

    # Compute q-criterion
    q = 0.5 * (omega_dot_omega - s_dot_s)

    # Pad outputs to match original shape
    pad_width = ((1, 1), (1, 1), (1, 1))  # Add 1 voxel on each side in x, y, z
    norm_mu = jnp.pad(norm_mu, pad_width, mode="constant", constant_values=0)
    q = jnp.pad(q, pad_width, mode="constant", constant_values=0)
    tau_xy = jnp.pad(tau_xy, pad_width, mode="constant", constant_values=0)
    tau_xz = jnp.pad(tau_xz, pad_width, mode="constant", constant_values=0)
    tau_yz = jnp.pad(tau_yz, pad_width, mode="constant", constant_values=0)
    tau_magnitude = jnp.pad(tau_magnitude, pad_width, mode="constant", constant_values=0)

    return norm_mu, q, tau_xy, tau_xz, tau_yz, tau_magnitude

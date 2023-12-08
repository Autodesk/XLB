import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
import numpy as np
from time import time
import pyvista as pv
from jax.image import resize
from jax import jit
import jax.numpy as jnp
from functools import partial
import trimesh

import os
import __main__


@partial(jit, static_argnums=(1, 2))
def downsample_field(field, factor, method='bicubic'):
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

def save_image(timestep, fld, prefix=None):
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
    This function saves the field as an image in the PNG format. The filename is based on the name of the main script file, the provided prefix, and the timestep number.
    If the field is 3D, the magnitude of the field is calculated and saved. The image is saved with the 'nipy_spectral' colormap and the origin set to 'lower'.
    """
    fname = os.path.basename(__main__.__file__)
    fname = os.path.splitext(fname)[0]
    if prefix is not None:
        fname = prefix + fname
    fname = fname + "_" + str(timestep).zfill(4)

    if len(fld.shape) > 3:
        raise ValueError("The input field should be 2D!")
    elif len(fld.shape) == 3:
        fld = np.sqrt(fld[..., 0] ** 2 + fld[..., 1] ** 2)

    plt.clf()
    plt.imsave(fname + '.png', fld.T, cmap=cm.nipy_spectral, origin='lower')

def save_fields_vtk(timestep, fields, output_dir='.', prefix='fields'):
    """
    Save VTK fields to the specified directory.

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

    Returns
    -------
    None

    Notes
    -----
    This function saves the VTK fields in the specified directory, with filenames based on the provided timestep number
    and the filename. For example, if the timestep number is 10 and the file name is fields, the VTK file
    will be saved as 'fields_0000010.vtk'in the specified directory.

    """
    # Assert that all fields have the same dimensions except for the last dimension assuming fields is a dictionary
    for key, value in fields.items():
        if key == list(fields.keys())[0]:
            dimensions = value.shape
        else:
            assert value.shape == dimensions, "All fields must have the same dimensions!"

    output_filename = os.path.join(output_dir, prefix +  "_" + f"{timestep:07d}.vtk")

    # Add 1 to the dimensions tuple as we store cell values
    dimensions = tuple([dim + 1 for dim in dimensions])

    # Create a uniform grid
    if value.ndim == 2:
        dimensions = dimensions + (1,)

    grid = pv.ImageData(dimensions=dimensions)

    # Add the fields to the grid
    for key, value in fields.items():
        grid[key] = value.flatten(order='F')

    # Save the grid to a VTK file
    start = time()
    grid.save(output_filename, binary=True)
    print(f"Saved {output_filename} in {time() - start:.6f} seconds.")

def live_volume_randering(timestep, field):
    # WORK IN PROGRESS
    """
    Live rendering of a 3D volume using pyvista.

    Parameters
    ----------
    field (np.ndarray): A 3D array containing the field to be rendered.

    Returns
    -------
    None

    Notes
    -----
    This function uses pyvista to render a 3D volume. The volume is rendered with a colormap based on the field values.
    The colormap is updated every 0.1 seconds to reflect changes to the field.

    """
    # Create a uniform grid (Note that the field must be 3D) otherwise raise error
    if field.ndim != 3:
        raise ValueError("The input field must be 3D!")
    dimensions = field.shape
    grid = pv.ImageData(dimensions=dimensions)

    # Add the field to the grid
    grid['field'] = field.flatten(order='F')

    # Create the rendering scene
    if timestep == 0:
        plt.ion()
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.title("Live rendering of the field")
        pl = pv.Plotter(off_screen=True)
        pl.add_volume(grid, cmap='nipy_spectral', opacity='sigmoid_10', shade=False)
        plt.imshow(pl.screenshot())

    else:
        pl = pv.Plotter(off_screen=True)
        pl.add_volume(grid, cmap='nipy_spectral', opacity='sigmoid_10', shade=False)
        # Update the rendering scene every 0.1 seconds
        plt.imshow(pl.screenshot())
        plt.pause(0.1)

def save_BCs_vtk(timestep, BCs, gridInfo,  output_dir='.'):
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
    if gridInfo['nz'] == 0:
        gridDimensions = (gridInfo['nx'] + 1, gridInfo['ny'] + 1, 1)
        fieldDimensions = (gridInfo['nx'], gridInfo['ny'], 1)
    else:
        gridDimensions = (gridInfo['nx'] + 1, gridInfo['ny'] + 1, gridInfo['nz'] + 1)
        fieldDimensions = (gridInfo['nx'], gridInfo['ny'], gridInfo['nz'])

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
        if gridInfo['dim'] == 2:
            bcIndices = np.ravel_multi_index(bcIndices, fieldDimensions[:-1], order='F')
        else:
            bcIndices = np.ravel_multi_index(bcIndices, fieldDimensions, order='F')

        grid[bcName] = np.zeros(fieldDimensions, dtype=bool).flatten(order='F')
        grid[bcName][bcIndices] = True

    # Save the grid to a VTK file
    output_filename = os.path.join(output_dir,  "BCs_" + f"{timestep:07d}.vtk")

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
    return tuple(jnp.rint(indices_rotated).astype('int32').T)

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


def axangle2mat(axis, angle, is_normalized=False):
    ''' Rotation matrix for rotation angle `angle` around `axis`
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
    '''
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
        [zxC - ys, yzC + xs, z * zC + c]])

@partial(jit)
def q_criterion(u):
    # Compute derivatives
    u_x = u[..., 0]
    u_y = u[..., 1]
    u_z = u[..., 2]

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
    norm_mu = jnp.sqrt(mu_x ** 2 + mu_y ** 2 + mu_z ** 2)

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
    s_dot_s = (
        s_0_0 ** 2 + s_0_1 ** 2 + s_0_2 ** 2 +
        s_1_0 ** 2 + s_1_1 ** 2 + s_1_2 ** 2 +
        s_2_0 ** 2 + s_2_1 ** 2 + s_2_2 ** 2
    )

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
        omega_0_0 ** 2 + omega_0_1 ** 2 + omega_0_2 ** 2 +
        omega_1_0 ** 2 + omega_1_1 ** 2 + omega_1_2 ** 2 +
        omega_2_0 ** 2 + omega_2_1 ** 2 + omega_2_2 ** 2
    )

    # Compute q-criterion
    q = 0.5 * (omega_dot_omega - s_dot_s)

    return norm_mu, q



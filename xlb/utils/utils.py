import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from time import time
import pyvista as pv
from jax.image import resize
from jax import jit
import jax.numpy as jnp
from functools import partial
import trimesh
import warp as wp

import os
import __main__
import importlib
from contextlib import nullcontext


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


def save_fields_vtk(fields, timestep, output_dir=".", prefix="fields"):
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
    # Assert that all fields have the same dimensions
    for key, value in fields.items():
        if key == list(fields.keys())[0]:
            dimensions = value.shape
        else:
            assert value.shape == dimensions, "All fields must have the same dimensions!"

    output_filename = os.path.join(output_dir, prefix + "_" + f"{timestep:07d}.vtk")

    # Add 1 to the dimensions tuple as we store cell values
    dimensions = tuple([dim + 1 for dim in dimensions])

    # Create a uniform grid
    if value.ndim == 2:
        dimensions = dimensions + (1,)

    grid = pv.ImageData(dimensions=dimensions)

    # Add the fields to the grid
    for key, value in fields.items():
        grid[key] = value.flatten(order="F")

    # Save the grid to a VTK file
    start = time()
    grid.save(output_filename, binary=True)
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


@wp.kernel
def get_color(
    low: float,
    high: float,
    values: wp.array(dtype=float),
    out_color: wp.array(dtype=wp.vec3),
):
    """
    Colorize scalars using a rainbow color map.

    Parameters
    ----------
    low : float
        The lower bound of the color map.
    high : float
        The upper bound of the color map.
    values : wp.array(dtype=float)
        The values to colorize.
    out_color : wp.array(dtype=wp.vec3)
        The output colors.

    Returns
    -------
    None
    """
    tid = wp.tid()
    v = values[tid]
    r = 1.0
    g = 1.0
    b = 1.0
    if v < low:
        v = low
    if v > high:
        v = high
    dv = high - low
    if v < (low + 0.25 * dv):
        r = 0.0
        g = 4.0 * (v - low) / dv
    elif v < (low + 0.5 * dv):
        r = 0.0
        b = 1.0 + 4.0 * (low + 0.25 * dv - v) / dv
    elif v < (low + 0.75 * dv):
        r = 4.0 * (v - low - 0.5 * dv) / dv
        b = 0.0
    else:
        g = 1.0 + 4.0 * (low + 0.75 * dv - v) / dv
        b = 0.0
    out_color[tid] = wp.vec3(r, g, b)


def colorize_scalars(scalars, device=None, value_range=None, percentiles=(5, 95), target=None):
    """
    Colorize scalars using a rainbow color map.

    Parameters
    ----------
    scalars : wp.array(dtype=float)
        The scalars to colorize.
    device : wp.Device, optional
        The device to use for the colorization.
    value_range : tuple, optional
        The value range to use for the colorization.
    percentiles : tuple, optional
        The percentiles to use for the colorization.
    target : wp.array(dtype=wp.vec3), optional
        The target array to store the colors.

    Returns
    -------
    wp.array(dtype=wp.vec3)
        The colors.
    tuple
        The value range used for the colorization.
    """
    if device is None:
        device = scalars.device
    colors = target if target is not None else wp.empty(scalars.shape[0], dtype=wp.vec3, device=device)
    if value_range is None:
        scalars_np = scalars.numpy()
        low = float(np.percentile(scalars_np, percentiles[0]))
        high = float(np.percentile(scalars_np, percentiles[1]))
    else:
        low = float(value_range[0])
        high = float(value_range[1])
    if abs(high - low) < 1e-6:
        high = low + 1e-6
    wp.launch(
        kernel=get_color,
        dim=scalars.shape[0],
        inputs=(low, high, scalars),
        outputs=(colors,),
        device=device,
    )
    return colors, (low, high)


def _normalize_clip_values(values):
    """
    Internal function to normalize clip values.

    Parameters
    ----------
    values : tuple, optional
        The clip values to normalize.

    Returns
    -------
    tuple
        The normalized clip values.
    """
    if values is None:
        return (0, 0, 0)
    if isinstance(values, (int, float)):
        v = int(values)
        return (v, v, v)
    seq = tuple(int(x) for x in values)
    if len(seq) != 3:
        raise ValueError("clip values must have length 3")
    return seq


def _slice_velocity_field(field, clip_lower, clip_upper):
    """
    Internal function to slice a velocity field.

    Parameters
    ----------
    field : wp.array(dtype=float)
        The velocity field to slice.
    clip_lower : tuple, optional
        The lower clip values.
    clip_upper : tuple, optional
        The upper clip values.

    Returns
    -------
    wp.array(dtype=float)
        The sliced velocity field.
    """
    lower = _normalize_clip_values(clip_lower)
    upper = _normalize_clip_values(clip_upper)
    slices = [slice(None)]
    for l, u in zip(lower, upper):
        start = l if l > 0 else None
        stop = -u if u > 0 else None
        slices.append(slice(start, stop))
    return field[tuple(slices)]


def _clone_to_device(array, device):
    """
    Internal function to clone an array to a device.

    Parameters
    ----------
    array : wp.array(dtype=float)
        The array to clone.
    device : wp.Device
        The device to clone the array to.

    Returns
    -------
    wp.array(dtype=float)
        The cloned array.
    """
    if hasattr(array, "device") and array.device == device:
        return array
    return wp.clone(array, device=device)


def _get_usd_modules():
    """
    Internal function to get the USD modules.

    Returns
    -------
    tuple
        The USD modules (UsdGeom, Vt).
    """
    UsdGeom = importlib.import_module("pxr.UsdGeom")
    Vt = importlib.import_module("pxr.Vt")
    return UsdGeom, Vt


def save_usd_vorticity(
    timestep,
    post_process_interval,
    bc_mask,
    f_current,
    grid_shape,
    usd_mesh,
    vorticity_operator,
    precision_policy,
    vorticity_threshold,
    usd_stage,
    device=None,
    clip_lower=None,
    clip_upper=None,
    color_percentiles=(5, 95),
    color_range=None,
):
    """
    Save the vorticity field to a USD mesh.

    Parameters
    ----------
    timestep : int
        The timestep.
    post_process_interval : int
        The post-process interval.
    bc_mask : wp.array(dtype=bool)
        The boundary mask.
    f_current : wp.array(dtype=float)
        The current field.
    grid_shape : tuple
        The shape of the grid.
    usd_mesh : pxr.Usd.Mesh
        The USD mesh to save the vorticity field to.
    vorticity_operator : xlb.operator.vorticity.VorticityOperator
        The vorticity operator.
    precision_policy : xlb.precision_policy.PrecisionPolicy
        The precision policy.
    vorticity_threshold : float
        The vorticity threshold.
    usd_stage : pxr.Usd.Stage
        The USD stage.
    device : wp.Device, optional
        The device to use for the computation.
    clip_lower : tuple, optional
        The lower clip values.
    clip_upper : tuple, optional
        The upper clip values.
    color_percentiles : tuple, optional
        The percentiles to use for the colorization.
    color_range : tuple, optional
        The value range to use for the colorization.

    Returns
    -------
    None
    """
    from xlb.compute_backend import ComputeBackend
    from xlb.operator.macroscopic import Macroscopic
    from xlb.operator.postprocess import GridToPoint
    import xlb

    if device is None:
        device = getattr(f_current, "device", "cpu")
    clip_lower = _normalize_clip_values(clip_lower)
    clip_upper = _normalize_clip_values(clip_upper)
    f_current_dev = _clone_to_device(f_current, device)
    bc_mask_dev = _clone_to_device(bc_mask, device)
    with wp.ScopedDevice(device):
        velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=ComputeBackend.WARP)
        macro_wp = Macroscopic(compute_backend=ComputeBackend.WARP, precision_policy=precision_policy, velocity_set=velocity_set)
        rho = wp.zeros((1, *grid_shape), dtype=wp.float32, device=device)
        u = wp.zeros((3, *grid_shape), dtype=wp.float32, device=device)
        rho, u = macro_wp(f_current_dev, rho, u)
        u = _slice_velocity_field(u, clip_lower, clip_upper)
        vorticity = wp.zeros((3, *u.shape[1:]), dtype=wp.float32, device=device)
        vorticity_magnitude = wp.zeros((1, *u.shape[1:]), dtype=wp.float32, device=device)
        vorticity, vorticity_magnitude = vorticity_operator(u, bc_mask_dev, vorticity, vorticity_magnitude)
        max_verts = grid_shape[0] * grid_shape[1] * grid_shape[2] * 5
        max_tris = grid_shape[0] * grid_shape[1] * grid_shape[2] * 3
        mc = wp.MarchingCubes(nx=u.shape[1], ny=u.shape[2], nz=u.shape[3], max_verts=max_verts, max_tris=max_tris, device=device)
        mc.surface(vorticity_magnitude[0], vorticity_threshold)
        if mc.verts.shape[0] == 0:
            print(f"Warning: No vertices found for vorticity at timestep {timestep}.")
            return
        grid_to_point_op = GridToPoint(precision_policy=precision_policy, compute_backend=ComputeBackend.WARP)
        scalars = wp.zeros(mc.verts.shape[0], dtype=wp.float32, device=device)
        scalars = grid_to_point_op(vorticity_magnitude, mc.verts, scalars)
        colors, value_range = colorize_scalars(
            scalars,
            device=device,
            value_range=color_range,
            percentiles=color_percentiles,
        )
        vertices = mc.verts.numpy()
        indices = mc.indices.numpy()
        colors_np = colors.numpy()
    tri_count = len(indices) // 3
    time_code = timestep // post_process_interval
    UsdGeom, _ = _get_usd_modules()
    usd_mesh.GetPointsAttr().Set(vertices.tolist(), time=time_code)
    usd_mesh.GetFaceVertexCountsAttr().Set([3] * tri_count, time=time_code)
    usd_mesh.GetFaceVertexIndicesAttr().Set(indices.tolist(), time=time_code)
    usd_mesh.GetDisplayColorAttr().Set(colors_np.tolist(), time=time_code)
    UsdGeom.Primvar(usd_mesh.GetDisplayColorAttr()).SetInterpolation("vertex")
    print(f"Vorticity visualization at timestep {timestep}:")
    print(f"  Number of vertices: {len(vertices)}")
    print(f"  Number of triangles: {tri_count}")
    print(f"  Vorticity range: [{value_range[0]:.6f}, {value_range[1]:.6f}]")


def save_usd_q_criterion(
    timestep,
    post_process_interval,
    bc_mask,
    f_current,
    grid_shape,
    usd_mesh,
    q_criterion_operator,
    precision_policy,
    q_threshold,
    usd_stage,
    device=None,
    clip_lower=None,
    clip_upper=None,
    color_range=(0.0, 0.1),
    color_percentiles=None,
):
    """
    Save the Q-criterion field to a USD mesh.

    Parameters
    ----------
    timestep : int
        The timestep.
    post_process_interval : int
        The post-process interval.
    bc_mask : wp.array(dtype=bool)
        The boundary mask.
    f_current : wp.array(dtype=float)
        The current field.
    grid_shape : tuple
        The shape of the grid.
    usd_mesh : pxr.Usd.Mesh
        The USD mesh to save the Q-criterion field to.
    q_criterion_operator : xlb.operator.q_criterion.QCriterionOperator
        The Q-criterion operator.
    precision_policy : xlb.precision_policy.PrecisionPolicy
        The precision policy.
    q_threshold : float
        The Q-criterion threshold.
    usd_stage : pxr.Usd.Stage
        The USD stage.
    device : wp.Device, optional
        The device to use for the computation.
    clip_lower : tuple, optional
        The lower clip values.
    clip_upper : tuple, optional
        The upper clip values.
    color_range : tuple, optional
        The value range to use for the colorization.
    color_percentiles : tuple, optional
        The percentiles to use for the colorization.

    Returns
    -------
    None
    """
    from xlb.compute_backend import ComputeBackend
    from xlb.operator.macroscopic import Macroscopic
    from xlb.operator.postprocess import GridToPoint
    import xlb

    if device is None:
        device = getattr(f_current, "device", "cpu")
    clip_lower = _normalize_clip_values(clip_lower)
    clip_upper = _normalize_clip_values(clip_upper)
    f_current_dev = _clone_to_device(f_current, device)
    bc_mask_dev = _clone_to_device(bc_mask, device)
    with wp.ScopedDevice(device):
        velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, compute_backend=ComputeBackend.WARP)
        macro_wp = Macroscopic(compute_backend=ComputeBackend.WARP, precision_policy=precision_policy, velocity_set=velocity_set)
        rho = wp.zeros((1, *grid_shape), dtype=wp.float32, device=device)
        u = wp.zeros((3, *grid_shape), dtype=wp.float32, device=device)
        rho, u = macro_wp(f_current_dev, rho, u)
        u = _slice_velocity_field(u, clip_lower, clip_upper)
        norm_mu = wp.zeros((1, *u.shape[1:]), dtype=wp.float32, device=device)
        q_field = wp.zeros((1, *u.shape[1:]), dtype=wp.float32, device=device)
        norm_mu, q_field = q_criterion_operator(u, bc_mask_dev, norm_mu, q_field)
        max_verts = grid_shape[0] * grid_shape[1] * grid_shape[2] * 5
        max_tris = grid_shape[0] * grid_shape[1] * grid_shape[2] * 3
        mc = wp.MarchingCubes(nx=u.shape[1], ny=u.shape[2], nz=u.shape[3], max_verts=max_verts, max_tris=max_tris, device=device)
        mc.surface(q_field[0], q_threshold)
        if mc.verts.shape[0] == 0:
            print(f"Warning: No vertices found for Q-criterion at timestep {timestep}.")
            return
        grid_to_point_op = GridToPoint(precision_policy=precision_policy, compute_backend=ComputeBackend.WARP)
        scalars = wp.zeros(mc.verts.shape[0], dtype=wp.float32, device=device)
        scalars = grid_to_point_op(norm_mu, mc.verts, scalars)
        if color_range is None:
            percentiles = color_percentiles if color_percentiles is not None else (5, 95)
            colors, used_range = colorize_scalars(
                scalars,
                device=device,
                percentiles=percentiles,
            )
        else:
            colors, used_range = colorize_scalars(
                scalars,
                device=device,
                value_range=color_range,
                percentiles=color_percentiles if color_percentiles is not None else (5, 95),
            )
        vertices = mc.verts.numpy()
        indices = mc.indices.numpy()
        colors_np = colors.numpy()
    tri_count = len(indices) // 3
    time_code = timestep // post_process_interval
    UsdGeom, _ = _get_usd_modules()
    usd_mesh.GetPointsAttr().Set(vertices.tolist(), time=time_code)
    usd_mesh.GetFaceVertexCountsAttr().Set([3] * tri_count, time=time_code)
    usd_mesh.GetFaceVertexIndicesAttr().Set(indices.tolist(), time=time_code)
    usd_mesh.GetDisplayColorAttr().Set(colors_np.tolist(), time=time_code)
    UsdGeom.Primvar(usd_mesh.GetDisplayColorAttr()).SetInterpolation("vertex")
    print(f"Q-criterion visualization at timestep {timestep}:")
    print(f"  Number of vertices: {len(vertices)}")
    print(f"  Number of triangles: {tri_count}")
    print(f"  Scalar range: [{used_range[0]:.6f}, {used_range[1]:.6f}]")


def update_usd_lagrangian_parts(
    timestep,
    post_process_interval,
    vertices_wp,
    parts,
    vertex_offset=None,
    lag_forces=None,
    force_component=0,
    device=None,
):
    """
    Update the USD lagrangian parts. The lagrangian parts are updated with the vertices and faces. The forces are colorized using a rainbow color map.
    The color map is a linear interpolation between the lower and upper bounds.
    Parameters
    ----------
    timestep : int
        The timestep.
    post_process_interval : int
        The post-process interval.
    vertices_wp : wp.array(dtype=float)
        The vertices of the lagrangian parts.
    parts : list
        The parts of the lagrangian mesh.
    vertex_offset : tuple, optional
        The vertex offset.
    lag_forces : wp.array(dtype=float), optional
        The forces of the lagrangian parts.
    force_component : int, optional
        The component of the forces to colorize.
    device : wp.Device, optional
        The device to use for the computation.

    Returns
    -------
    None
    """
    vertices_np = vertices_wp.numpy()
    if vertex_offset is not None:
        offset_array = np.asarray(vertex_offset, dtype=np.float64)
        if offset_array.ndim == 0:
            offset_array = np.repeat(offset_array, 3)
        vertices_np = vertices_np - offset_array
    time_code = timestep // post_process_interval
    lag_forces_np = lag_forces.numpy() if lag_forces is not None else None
    if device is None:
        device = getattr(vertices_wp, "device", None)
    if device is None:
        device = "cpu"
    UsdGeom, Vt = _get_usd_modules()
    for part in parts:
        start = int(part.get("start", 0))
        end = int(part.get("end", vertices_np.shape[0]))
        faces = np.asarray(part["faces"], dtype=np.int32)
        if part.get("shift_indices", True):
            faces = faces - start
        usd_mesh = part["usd_mesh"]
        part_vertices = vertices_np[start:end]
        tri_count = len(faces)
        usd_mesh.GetPointsAttr().Set(part_vertices.tolist(), time=time_code)
        usd_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * tri_count), time=time_code)
        usd_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.flatten().tolist()), time=time_code)
        if lag_forces_np is not None and part.get("colorize", False):
            component = int(part.get("force_component", force_component))
            forces_np = lag_forces_np[start:end, component]
            context = wp.ScopedDevice(device) if device is not None else nullcontext()
            with context:
                force_wp = wp.from_numpy(forces_np.astype(np.float32), dtype=wp.float32, device=device)
                colors_wp = wp.zeros(part_vertices.shape[0], dtype=wp.vec3, device=device)
                if part.get("color_percentiles") is not None:
                    colors_wp, used_range = colorize_scalars(
                        force_wp,
                        device=device,
                        percentiles=part["color_percentiles"],
                        target=colors_wp,
                    )
                else:
                    value_range = part.get("color_range")
                    if value_range is None:
                        low = float(np.min(forces_np))
                        high = float(np.max(forces_np))
                        if abs(high - low) < 1e-6:
                            high = low + 1e-6
                        value_range = (low, high)
                    colors_wp, used_range = colorize_scalars(
                        force_wp,
                        device=device,
                        value_range=value_range,
                        target=colors_wp,
                    )
                colors_np = colors_wp.numpy()
            usd_mesh.GetDisplayColorAttr().Set(colors_np.tolist(), time=time_code)
            UsdGeom.Primvar(usd_mesh.GetDisplayColorAttr()).SetInterpolation("vertex")
    print(f"Lagrangian meshes updated at timestep {timestep}")


def plot_object_placement(vertices_wp, grid_shape, filename, title, object_label="Object"):
    """
    Plot the object placement.
    The object placement is plotted as a polygon in the domain. The domain is the bounding box of the vertices.
    The plot is saved as a PNG file.

    Parameters
    ----------
    vertices_wp : wp.array(dtype=float)
        The vertices of the object.
    grid_shape : tuple
        The shape of the grid.
    filename : str
        The filename to save the plot to.
    title : str
        The title of the plot.
    object_label : str, optional
        The label of the object.

    Returns
    -------
    None
    """
    verts = vertices_wp.numpy()
    obj_min = verts.min(axis=0)
    obj_max = verts.max(axis=0)
    plt.figure(figsize=(10, 5))
    domain_x = [0, grid_shape[0], grid_shape[0], 0, 0]
    domain_y = [0, 0, grid_shape[1], grid_shape[1], 0]
    plt.plot(domain_x, domain_y, "k-", linewidth=1, label="Domain")
    poly_x = [obj_min[0], obj_max[0], obj_max[0], obj_min[0], obj_min[0]]
    poly_y = [obj_min[1], obj_min[1], obj_max[1], obj_max[1], obj_min[1]]
    plt.plot(poly_x, poly_y, "r-", linewidth=2, label=object_label)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

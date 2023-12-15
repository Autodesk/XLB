import warp as wp
import cupy as cp
import jax.dlpack as jdlpack
import jax


def _cupy_to_backend(cupy_array, backend):
    """
    Convert cupy array to backend array

    Parameters
    ----------
    cupy_array : cupy.ndarray
        Input cupy array
    backend : str
        Backend to convert to. Options are "jax", "warp", or "cupy"
    """

    # Convert cupy array to backend array
    dl_array = cupy_array.toDlpack()
    if backend == "jax":
        backend_array = jdlpack.from_dlpack(dl_array)
    elif backend == "warp":
        backend_array = wp.from_dlpack(dl_array)
    elif backend == "cupy":
        backend_array = cupy_array
    else:
        raise ValueError(f"Backend {backend} not supported")
    return backend_array


def _backend_to_cupy(backend_array, backend):
    """
    Convert backend array to cupy array

    Parameters
    ----------
    backend_array : backend.ndarray
        Input backend array
    backend : str
        Backend to convert from. Options are "jax", "warp", or "cupy"
    """

    # Convert backend array to cupy array
    if backend == "jax":
        (jax.device_put(0.0) + 0).block_until_ready()
        dl_array = jdlpack.to_dlpack(backend_array)
    elif backend == "warp":
        dl_array = wp.to_dlpack(backend_array)
    elif backend == "cupy":
        return backend_array
    else:
        raise ValueError(f"Backend {backend} not supported")
    cupy_array = cp.fromDlpack(dl_array)
    return cupy_array


def _stream_to_backend(stream, backend):
    """
    Convert cupy stream to backend stream

    Parameters
    ----------
    stream : cupy.cuda.Stream
        Input cupy stream
    backend : str
        Backend to convert to. Options are "jax", "warp", or "cupy"
    """

    # Convert stream to backend stream
    if backend == "jax":
        raise ValueError("Jax currently does not support streams")
    elif backend == "warp":
        backend_stream = wp.Stream(cuda_stream=stream.ptr)
    elif backend == "cupy":
        backend_stream = stream
    else:
        raise ValueError(f"Backend {backend} not supported")
    return backend_stream

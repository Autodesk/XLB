# Installation

## Getting Startedd
To get started with XLB, you can install it using pip. There are different installation options depending on your hardware and needs:

### Basic Installation (CPU-only)
```bash
pip install xlb
```

### Installation with CUDA support (for NVIDIA GPUs)
This installation is for the JAX backend with CUDA support:
```bash
pip install "xlb[cuda]"
```

### Installation with TPU support
This installation is for the JAX backend with TPU support:
```bash
pip install "xlb[tpu]"
```

### Notes:
- For Mac users: Use the basic CPU installation command as JAX's GPU support is not available on MacOS
- The NVIDIA Warp backend is included in all installation options and supports CUDA automatically when available
- The installation options for CUDA and TPU only affect the JAX backend

To install the latest development version from source:

```bash
pip install git+https://github.com/Autodesk/XLB.git
```

The changelog for the releases can be found [here](https://github.com/Autodesk/XLB/blob/main/CHANGELOG.md).

For examples to get you started please refer to the [examples](https://github.com/Autodesk/XLB/tree/main/examples) folder.
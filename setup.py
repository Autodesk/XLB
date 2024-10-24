from setuptools import setup, find_packages

setup(
    name="xlb",
    version="0.2.1",
    description="XLB: Accelerated Lattice Boltzmann (XLB) for Physics-based ML",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="XLB Team",
    url="https://github.com/Autodesk/XLB",
    license="Apache License 2.0",
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.9.2",
        "numpy>=2.1.2",
        "pyvista>=0.44.1",
        "trimesh>=4.4.9",
        "warp-lang>=1.4.0",
        "numpy-stl>=3.1.2",
        "pydantic>=2.9.1",
        "ruff>=0.6.5",
        "jax>=0.4.34",  # Base JAX CPU-only requirement
    ],
    extras_require={
        "cuda": ["jax[cuda12]>=0.4.34"],  # For CUDA installations
        "tpu": ["jax[tpu]>=0.4.34"],  # For TPU installations
    },
    python_requires=">=3.10",
    dependency_links=["https://storage.googleapis.com/jax-releases/libtpu_releases.html"],
)

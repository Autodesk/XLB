from setuptools import setup, find_packages

setup(
    name="xlb",
    version="0.3.0",
    description="XLB: Accelerated Lattice Boltzmann (XLB) for Physics-based ML",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mehdi Ataei",
    url="https://github.com/Autodesk/XLB",
    license="Apache License 2.0",
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.9.2",
        "numpy>=2.1.2",
        "pyvista>=0.44.1",
        "trimesh>=4.4.9",
        "warp-lang>=1.10.0",
        "numpy-stl>=3.1.2",
        "pydantic>=2.9.1",
        "ruff>=0.14.1",
        "jax>=0.8.0",  # Base JAX CPU-only requirement
    ],
    extras_require={
        "cuda": ["jax[cuda13]>=0.8.0"],  # For CUDA installations
        "tpu": ["jax[tpu]>=0.8.0"],  # For TPU installations
    },
    python_requires=">=3.11",
    dependency_links=["https://storage.googleapis.com/jax-releases/libtpu_releases.html"],
)

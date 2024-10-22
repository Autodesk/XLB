from setuptools import setup, find_packages

setup(
    name='xlb',
    version='0.2.0',
    description='XLB: Accelerated Lattice Boltzmann (XLB) for Physics-based ML',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Mehdi Ataei',
    url='https://github.com/Autodesk/XLB',
    license='Apache License 2.0',
    packages=find_packages(),
    install_requires=[
        'jax[cuda]>=0.4.34',
        'matplotlib>=3.9.2',
        'numpy>=2.1.2',
        'pyvista>=0.44.1',
        'trimesh>=4.4.9',
        'warp-lang>=1.4.0',
        'numpy-stl>=3.1.2',
        'pydantic>=2.9.1',
        'ruff>=0.6.5'
    ],
    python_requires='>=3.10',
)

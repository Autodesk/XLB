# Performance Comparisons

This directory contains a minimal LBM implementation in Warp, Numba, and Jax. The
code can be run with the following command:

```bash
python3 lattice_boltzmann.py
```

This will give MLUPs numbers for each implementation. The Warp implementation
is the fastest, followed by Numba, and then Jax.

This example should be used as a test for properly implementing more backends in
XLB.

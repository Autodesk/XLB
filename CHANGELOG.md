# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- _No changes yet_  <!-- Placeholder for future changes -->

## [0.3.2] - 2026-11-10

### Added
- Multi-resolution grid refinement and Neon backend support (#159)
- Differentiable LBM example for inverse problem solving with JAX (#156)

### Changed
- Upgraded Warp dependency to 1.19 (#168)
- PyPI builds omit the `[neon]` extra (direct-URL `neon_gpu` requirement); Neon remains available from source/git installs

### Removed
- Deprecated functions (#154)

## [0.2.1] - 2024-12-05

### Fixed
- mkdocs is now configured correctly for the new project structure
- JAX installation is now handled correctly for different configurations (CPU, CUDA, TPU)
- Fixed a couple of bugs in 2D regularied_bc and kbc (Warp) that emerged after merging 2d and 3d kernels

### Added

- Added abstraction layer for boundary condition efficient encoding/decoding of auxiliary data
- Added the capability to add profiles to boundary conditions
- Added prepare_fields method to the Stepper class to allow for more automatic preparation of fields

## [0.2.0] - 2024-10-22

### Added
- XLB is now installable via pip
- Complete rewrite of the codebase for better modularity and extensibility based on "Operators" design pattern
- Added NVIDIA's Warp backend for state-of-the-art performance
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### Types of changes

- `Added` for new features.
- `Changed` for changes in existing functionality.
- `Deprecated` for soon-to-be removed features.
- `Removed` for now removed features.
- `Fixed` for any bug fixes.
- `Security` in case of vulnerabilities.

## [Unreleased]

### Added

- A simple jupyter notebook (#55).
- `sdf_bias` as an alternative way for SDF initialization in `implicit-volume` (#57).

### Changed

- Remove `trainer` from the constructor arguments of prompt processors (#56).
- Use a reparametrization trick for the SDS loss (#57).

## [v0.1.0]

### Added

- Implementation of DreamFusion, Magic3D, SJC, Latent-NeRF and Sketch-Shape.
- Implementation of the geometry stage of Fantasia3D.
- Multi-GPU training support (#33).
- Mesh export, supporting obj with mtl and obj with vertex colors (#44).

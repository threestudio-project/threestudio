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
- Automatically remove outliers with a small number of faces when extracting surfaces (#61).
- The implementation of ProlificDreamer (#74, #105).
- An experimental implementation of using Zero-1-to-3 for 3D generation from a single image (#71).
- Support mesh initialization for `implicit-sdf` (#90).
- Easy-to-use geometry conversion by `system.geometry_convert_from`. This is used in the Magic3D and ProlificDreamer system and may inspire applications connecting multiple systems/algorithms (#105).
- Support prompt debiasing and manual assignment of view-dependent prompts (#98).
- The implementation of Perp-Neg (#98).
- Support patch-based renderer (#154).
- Support 3D reconstruction from multi-view images and 3D editing based on InstructNeRF2NeRF/ControlNet (#119).
- Support NeuS/VolSDF volume renderer and the coarse stage of TextMesh (#162,#121).
- Gradio web interface (#183).

### Changed

- Remove `trainer` from the constructor arguments of prompt processors (#56).
- Use a reparametrization trick for the SDS loss (#57).
- Make Magic3D coarse stage use analytic normal and orientation loss.
- Move the logic of getting text embeddings according to camera settings from prompt processors to guidance (#77).
- Remove `from_coarse` from the Magic3D system. Use `system.geometry_convert_from` instead (#105).

### Fixed

- Fix errors caused by empty rays (#152).

## [v0.1.0]

### Added

- Implementation of DreamFusion, Magic3D, SJC, Latent-NeRF and Sketch-Shape.
- Implementation of the geometry stage of Fantasia3D.
- Multi-GPU training support (#33).
- Mesh export, supporting obj with mtl and obj with vertex colors (#44).

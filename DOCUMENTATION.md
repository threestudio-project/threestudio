## Common Configurations

| name          | type          | description                                                                                                                                                                                                                                                           |
| ------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| name          | str           | Name of the experiment. Default: "default"                                                                                                                                                                                                                            |
| description   | str           | Description of the experiment. Default: ""                                                                                                                                                                                                                            |
| tag           | str           | Tag of the experiment. Default: ""                                                                                                                                                                                                                                    |
| seed          | str           | Global seed of the experiment. Used by `seed_everything` of PyTorch-Lightning. Default: 0                                                                                                                                                                             |
| use_timestamp | bool          | Whether to use the current timestamp as the suffix of the tag. Default: True                                                                                                                                                                                          |
| timestamp     | Optional[str] | The timestamp as the suffix of the tag. DO NOT set this manually. Default: None                                                                                                                                                                                       |
| exp_root_dir  | str           | The root directory for outputs of all the experiments. Default: "outputs"                                                                                                                                                                                             |
| exp_dir       | str           | The directory for outputs of the current experiment. DO NOT set this manually. It will be automatically set to `[exp_root_dir]/[name]`.                                                                                                                               |
| trial_name    | str           | Name of the trial. DO NOT set this manually. It will be automatically set to `[tag]@[timestamp]`.                                                                                                                                                                     |
| trial_dir     | str           | The directory for outputs for the current trial. DO NOT set this manually. It will be automatically set to `[exp_root_dir]/[name]/[trial_name].`                                                                                                                      |
| resume        | Optional[str] | The path to the checkpoint file to resume from. Default: None                                                                                                                                                                                                         |
| data_type     | str           | Type of the data module used. See [here](https://github.com/threestudio-project/threestudio/blob/main/DOCUMENTATION.md#data) for supported data modules. Default: ""                                                                                                  |
| data          | dict          | Configurations of the data module. Default: {}                                                                                                                                                                                                                        |
| system_type   | str           | Type of the system used. See [here](https://github.com/threestudio-project/threestudio/blob/main/DOCUMENTATION.md#systems) for supported systems. Default: ""                                                                                                         |
| system        | dict          | Configurations of the system. Defaut: {}                                                                                                                                                                                                                              |
| trainer       | dict          | Configurations of PyTorch-Lightning Trainer. See https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api for supported arguments. Exceptions: `logger` and `callbacks` are set in `launch.py`. Default: {}                                     |
| checkpoint    | dict          | Configurations of PyTorch-Lightning ModelCheckpoint callback, which defines when the checkpoint will be saved. See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#modelcheckpoint for supported arguments. Default: {} |

## Data

### random-camera-datamodule

| name                   | type                  | description                                                                                                                                                                                                                                                                                              |
| ---------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| height                 | Union[int, List[int]] | Height of the rendered image in training, should be an integer or a list of integers. If a list of integers, the training height will change according to `resolution_milestones`. Default: 64                                                                                                           |
| width                  | Union[int, List[int]] | Width of the rendered image in training, should be an integer or a list of integers. If a list of integers, the training width will change according to `resolution_milestones`. Default: 64                                                                                                             |
| batch_size             | Union[int, List[int]] | Number of images per batch in training. If a list of integers, the batch_size will change according to `resolution_milestones`. Default: 1                                                                                                                                                               |
| resolution_milestones  | List[int]             | The steps where the training resolution will change, must be in ascending order and in the length of `len(height) - 1`. Default: []                                                                                                                                                                      |
| eval_height            | int                   | Height of the rendered image in validation/testing. Default: 512                                                                                                                                                                                                                                         |
| eval_width             | int                   | Width of the rendered image in validation/testing. Default: 512                                                                                                                                                                                                                                          |
| eval_batch_size        | int                   | Number of images per batch in validation/testing. DO NOT change this. Default: 1                                                                                                                                                                                                                         |
| elevation_range        | Tuple[float,float]    | Camera elevation angle range to sample from in training, in degrees. Default: (-10,90)                                                                                                                                                                                                                   |
| azimuth_range          | Tuple[float,float]    | Camera azimuth angle range to sample from in training, in degrees. Default: (-180,180)                                                                                                                                                                                                                   |
| camera_distance_range  | Tuple[float,float]    | Camera distance range to sample from in training. Default: (1,1.5)                                                                                                                                                                                                                                       |
| fovy_range             | Tuple[float,float]    | Camera field of view (FoV) range along the y direction (vertical direction) to sample from in training, in degrees. Default: (40,70)                                                                                                                                                                     |
| camera_perturb         | float                 | Random perturbation ratio for the sampled camera positions in training. The sampled camera positions will be perturbed by `N(0,1) * camera_perturb`. Default: 0.1                                                                                                                                        |
| center_perturb         | float                 | Random perturbation ratio for the look-at point of the cameras in training. The look-at point wil be `N(0,1) * center_perturb`. Default: 0.2                                                                                                                                                             |
| up_perturb             | float                 | Random pertubation ratio for the up direction of the cameras in training. The up direction will be `[0,0,1] + N(0,1) * up_perturb`. Default: 0.02                                                                                                                                                        |
| light_position_perturb | float                 | Used to get random light directions from camera positions, only used when `light_sample_strategy="dreamfusion"`. The camera positions will be perturbed by `N(0,1) * light_position_perturb`, then the perturbed positions are used to determine the light directions. Default: 1.0                      |
| light_distance_range   | Tuple[float,float]    | Point light distance range to sample from in training. Default: (0.8,1.5)                                                                                                                                                                                                                                |
| eval_elevation_deg     | float                 | Camera elevation angle in validation/testing, in degrees. Default: 150                                                                                                                                                                                                                                   |
| eval_camera_distance   | float                 | Camera distance in validation/testing. Default: 15                                                                                                                                                                                                                                                       |
| eval_fovy_deg          | float                 | Camera field of view (FoV) along the y direction (vertical direction) in validation/testing, in degrees. Default: 70                                                                                                                                                                                     |
| light_sample_strategy  | str                   | Strategy to sample point light positions in training, in ["dreamfusion", "magic3d"]. "dreamfusion" uses strategy described in the DreamFusion paper; "magic3d" uses strategy decribed in the Magic3D paper. Default: "dreamfusion"                                                                       |
| batch_uniform_azimuth  | bool                  | Whether to ensure the uniformity of sampled azimuth angles in training as described in the Fantasia3D paper. If True, the `azimuth_range` is equally divided into `batch_size` bins and the azimuth angles are sampled from every bins. Default: True                                                    |
| progressive_until      | int                   | Number of iterations until which to progressively (linearly) increase elevation_range and azimuth_range from [`eval_elevation_deg`, `eval_elevation_deg`] and `[0.0, 0.0]`, to those values specified in `elevation_range` and `azimuth_range`. 0 means the range does not linearly increase. Default: 0 |

## Systems

Systems contain implementation of training/validation/testing logic for different methods.

**Common configurations for systems**

| name                          | type                | description                                                                                                                  |
| ----------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| loss                          | dict                | Dict that contains loss-related configurations. Default: {}                                                                  |
| optimizer                     | dict                | Optimizer configurations. Default: {}                                                                                        |
| scheduler                     | Optional[dict]      | Learning rate scheduler configurations. If None, does not use a scheduler. Default: None                                     |
| weights                       | Optional[str]       | Path to the weights to be loaded. This is different from `resume` in that this does not resume training state. Default: None |
| weights_ignore_modules        | Optional[List[str]] | List of modules that should be ignored when loading weights. Default: None                                                   |
| cleanup_after_validation_step | bool                | Whether to empty cache after each validation step. This will slow down validation. Default: False                            |
| cleanup_after_test_step       | bool                | Whether to empty cache after each test step. This will slow down testing. Default: False                                     |

Currently all implemented systems inherit to `BaseLift3DSystem`, which has the following common configurations:

| name                             | type          | description                                                                                                                                                                                                                       |
| -------------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| geometry_type                    | str           | Type of the geometry used in the system. See [here](https://github.com/threestudio-project/threestudio/blob/main/DOCUMENTATION.md#geometry) for supported geometry.                                                               |
| geometry                         | dict          | Configurations of the geometry.                                                                                                                                                                                                   |
| geometry_convert_from            | Optional[str] | The path to a checkpoint from which the geometry is converted. If not None, initialize the geometry from the specified source model. Default: None                                                                                |
| geometry_convert_override        | dict          | Configurations to override when initializing from a source geometry, only used when `geometry_convert_from` is specified. A typical use case is to specify an isosurface threshold value. Default: {}                             |
| geometry_convert_inherit_texture | bool          | Whether to load the encoding and feature network from the source geometry during conversion, only used when `geometry_convert_from` is specified. Default: False                                                                  |
| material_type                    | str           | Type of the material used in the system. See [here](https://github.com/threestudio-project/threestudio/blob/main/DOCUMENTATION.md#material) for supported materials.                                                              |
| matrial                          | dict          | Configurations of the material.                                                                                                                                                                                                   |
| background_type                  | str           | Type of the background used in the system. See [here](https://github.com/threestudio-project/threestudio/blob/main/DOCUMENTATION.md#background) for supported background.                                                         |
| background                       | dict          | Configurations of the background.                                                                                                                                                                                                 |
| renderer_type                    | str           | Type of the renderer used in the system. See [here](https://github.com/threestudio-project/threestudio/blob/main/DOCUMENTATION.md#renderers) for supported renderers.                                                             |
| renderer                         | dict          | Configurations of the renderer.                                                                                                                                                                                                   |
| guidance_type                    | str           | Type of the guidance used in the system. See [here](https://github.com/threestudio-project/threestudio/blob/main/DOCUMENTATION.md#guidance) for supported guidance.                                                               |
| guidance                         | dict          | Configurations of the guidance.                                                                                                                                                                                                   |
| prompt_processor_type            | str           | Type of the prompt processor used in the system. See [here](https://github.com/threestudio-project/threestudio/blob/main/DOCUMENTATION.md#prompt-processors) for supported prompt processors.                                     |
| prompt_processor                 | dict          | Configurations of the prompt processor.                                                                                                                                                                                           |
| exporter_type                    | str           | Type of the exporter used in the system. Only used in export stage. See [here](https://github.com/threestudio-project/threestudio/blob/main/DOCUMENTATION.md#prompt-processors) for supported exporters. Default: "mesh-exporter" |
| exporter                         | dict          | Configurations of the exporter.                                                                                                                                                                                                   |

### dreamfusion-system

This system has all the common configurations.

### magic3d-system

This system has all the common configurations, along with the following unique configurations:

| name       | type | description                                                                       |
| ---------- | ---- | --------------------------------------------------------------------------------- |
| refinement | bool | Whether to perform refinement (second stage in the Magic3D paper). Default: False |

### sjc-system

This system has all the common configurations, along with the following unique configurations:
| name               | type | description                                                                                                                                 |
| ------------------ | ---- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| subpixel_rendering | bool | Whether to perform subpixel rendering in validation/testing, which decodes a `128x128` latent feature map instead of `64x64`. Default: True |

### latentnerf-system

This system has all the common configurations, along with the following unique configurations:

| name        | type          | description                                                                      |
| ----------- | ------------- | -------------------------------------------------------------------------------- |
| refinement  | bool          | Whether to perform RGB space refinement. Default: False                          |
| guide_shape | Optional[str] | Path to the .obj file as the shape guidance, used in Sketch-Shape. Default: None |

### fantasia3d-system

This system has all the common configurations, along with the following unique configurations:

| name         | type | description                                                                                                                                                                                                                                                                                                                    |
| ------------ | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| latent_steps | int  | Number of steps for geometry optimization in latent space. In the first `latent_steps` steps, low resolution normal and mask are concatenated and fed to the latent diffusion model. After this high resolution normal is used to perform RGB space optimziation. Details are described in the Fantasia3D paper. Default: 2500 |
| texture      | bool | Whether to perform texture training. Default: False                                                                                                                                                                                                                                                                            |

### prolificdreamer-system

This system has all the common configurations, along with the following unique configurations:

| name              | type | description                                                                                            |
| ----------------- | ---- | ------------------------------------------------------------------------------------------------------ |
| stage             | str  | The training stage, in ["coarse", "geometry", "texture"]. Default: "coarse"                            |
| visualize_samples | bool | Whether to visualize samples of the pretrained and LoRA diffusion models in validation. Default: False |

## Geometry

Geometry models properties for locations in space, including density, SDF, feature and normal.

**Common configurations for implicit geometry**

| name                                 | type              | description                                                                                                                                                                                                                                                                      |
| ------------------------------------ | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| radius                               | float             | Half side length of the scene bounding box. Default: 1.0                                                                                                                                                                                                                         |
| isosurface                           | bool              | Whether to enable surface extraction. Default: True                                                                                                                                                                                                                              |
| isosusrface_method                   | str               | Method for surface extraction, in ["mc", "mt"]. "mc" uses the marching cubes algorithm, not differentiable; "mt" uses the marching tetrahedra algorithm, differentiable. Default: "mt"                                                                                           |
| isosurface_resolution                | int               | Grid resolution for surface extraction. Default: 128                                                                                                                                                                                                                             |
| isosurface_threshold                 | Union[float,str]  | The threshold value to determine the surface location of the implicit field, in [float, "auto"]. If "auto", use the mean value of the field as the threshold. Default: 0                                                                                                         |
| isosurface_chunk                     | int               | Chunk size when computing the field value on grid vertices, used to prevent OOM. If 0, does not use chunking. Default: 0                                                                                                                                                         |
| isosurface_coarse_to_fine            | bool              | Whether to extract the surface in a coarse-to-fine manner. If True, will first extract a coarse surface to get a tight bounding box, which is then used to extract a fine surface. Default: True                                                                                 |
| isosurface_deformable_grid           | bool              | Whether to optimize positions of grid vertices for surface extraction. Only support `isosurface_method=mt`. Default: False                                                                                                                                                       |
| isosurface_remove_outliers           | bool              | Whether to remove outlier components according to the number of faces. Only remove if the isosurface process does not require gradient. Default: True                                                                                                                            |
| isosurface_outlier_n_faces_threshold | Union[int, float] | Extracted mesh components with number of faces less than this threshold will be removed if `isosurface_remove_outliers=True`. If `int`, direcly used as the threshold number of faces; if `float`, used as the ratio of all face numbers to compute the threshold. Default: 0.01 |

### implicit-volume

| name                         | type             | description                                                                                                                                                                                                                                                                                                                     |
| ---------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| n_input_dims                 | int              | Number of input dimensions. Default: 3 (xyz)                                                                                                                                                                                                                                                                                    |
| n_feature_dims               | int              | Number of dimensions for the output features. Note that this should be aligned with the material used. Default: 3 (albedo)                                                                                                                                                                                                      |
| density_activation           | str              | Density activation function. See `get_activation` in `utils/ops.py` for all supported activation functions. Default: "softplus"                                                                                                                                                                                                 |
| density_bias                 | Union[float,str] | Offset value to be added to the pre-activated density, in [float, "blob_dreamfusion", "blob_magic3d"]. If "blob_dreamfusion", uses the blob density bias proposed in DreamFusion; if "blob_magic3d", uses the blob density bias proposed in Magic3D. Default: "blob_magic3d"                                                    |
| density_blob_scale           | float            | Controls the magnitude of the blob density if `density_bias` in ["blob_dreamfusion", "blob_magic3d"]. Default: 10                                                                                                                                                                                                               |
| density_blob_std             | float            | Controls the divergence of the blob density if `density_bias` in ["blob_dreamfusion", "blob_magic3d"]. Default: 0.5                                                                                                                                                                                                             |
| pos_encoding_config          | dict             | Configurations for the positional encoding. See https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#encodings for supported arguments. Default: {}                                                                                                                                                              |
| mlp_network_config           | dict             | Configurations for the MLP head for geometry attribute prediction (density, feature ...). See https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#networks for supported arguments. Default: {}                                                                                                                 |
| normal_type                  | str              | How the normal is computed, in ["analytic", "finite_difference", "pred"]. If "analytic", uses PyTorch auto-differentiation to compute the analytic normal; if "finite_difference", uses finite difference to compute the approximate normal; if "pred", uses an MLP network to predict the normal. Default: "finite_difference" |
| finite_difference_normal_eps | float            | The small epsilon value in finite difference to estimate the normal, used when `normal_type="finite_difference"`. Default: 0.01                                                                                                                                                                                                 |
| isosurface_threshold         | Union[float,str] | Inherit from common configurations, but default to "auto". Default: "auto"                                                                                                                                                                                                                                                      |

### implicit-sdf

| name                         | type                | description                                                                                                                                                                                                                                                                      |
| ---------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| n_input_dims                 | int                 | Number of input dimensions. Default: 3 (xyz)                                                                                                                                                                                                                                     |
| n_feature_dims               | int                 | Number of dimensions for the output features. Note that this should be aligned with the material used. Default: 3 (albedo)                                                                                                                                                       |
| pos_encoding_config          | dict                | Configurations for the positional encoding. See https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#encodings for supported arguments. Default: {}                                                                                                               |
| mlp_network_config           | dict                | Configurations for the MLP head for geometry attribute prediction (sdf, feature ...). See https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#networks for supported arguments. Default: {}                                                                      |
| normal_type                  | str                 | How the normal is computed, in ["finite_difference", "pred"]. If "finite_difference", uses finite difference to compute the approximate normal; if "pred", uses an MLP network to predict the normal. Default: "finite_difference"                                               |
| finite_difference_normal_eps | float               | The small epsilon value in finite difference to estimate the normal, used when `normal_type="finite_difference"`. Default: 0.01                                                                                                                                                  |
| shape_init                   | Optional[str]       | The shape to initializa the SDF as, in [None, "sphere", "ellipsoid"]. If None, does not initialize; if "sphere", initialized as a sphere; if "ellipsoid", initialized as an ellipsoid. Default: None                                                                             |
| shape_init_params            | Optional[Any]       | Parameters to specify the SDF initialization. If `shape_init="sphere"`, a float is used for the sphere radius; if `shape_init="ellipsoid"`, a tuple of three floats is used for the radius along x/y/z axis. Default: None                                                       |
| force_shape_init             | bool                | Whether to force initialization of the SDf even if weights are provided. Default:False                                                                                                                                                                                           |
| sdf_bias                     | Optional[float,str] | Bias value to be added to the network output SDF, in [float, "sphere", "ellipsoid"]. If "sphere", the SDF of a sphere is added; if "ellipsoid", the pseudo SDF of an ellipsoid is added. This can be used for SDF initialization as an alternative to `shape_init`. Default: 0.0 |
| sdf_bias_params              | Optional[Any]       | Parameters to specify the SDF initialization based on `sdf_bias`. If `sdf_bias="sphere"`, a float is used for the sphere radius; if `sdf_bias="ellipsoid"`, a tuple of three floats is used for the radius along x/y/z axis. Default: None                                       |

### volume-grid

An explicit geometry parameterized with a feature volume. The feature volume has a shape of `(n_feature_dims + 1) x grid_size`, one channel for density and the rest for material. The density is first scaled, then biased and finally activated.

| name                 | type                 | description                                                                                                                                                            |
| -------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| grid_size            | tuple[int, int, int] | The resolution of the feature volume. Default: (100, 100, 100)                                                                                                         |
| n_feature_dims       | int                  | The feature dimensions for its material. Default: 3                                                                                                                    |
| density_activation   | Optional[str]        | The activation to get the density value. Default: "softplus"                                                                                                           |
| density_bias         | Union[float, str]    | The initialization of the density. A float value indicates uniform initialization and `blob` indicates a ball centered at the center. Default: "blob"                  |
| density_blob_scale   | float                | The parameter for blob initialization. Default: 5.0                                                                                                                    |
| density_blob_std     | float                | The parameter for blob initialization. Default: 0.5                                                                                                                    |
| normal_type          | Optional[str]        | The way to compute the normal from density. If set to "pred", the normal is produced with another volume in the shape of `3 x grid_size`. Default: "finite_difference" |
| isosurface_threshold | Union[float,str]     | Inherit from common configurations, but default to "auto". Default: "auto"                                                                                             |

**Common configurations for explicit geometry**

| name                | type | description                                                                                                                                                                   |
| ------------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| pos_encoding_config | dict | Configurations for the positional encoding. See https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#encodings for supported arguments. Default: {}            |
| mlp_network_config  | dict | Configurations for the MLP head for feature prediction. See https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#networks for supported arguments. Default: {} |
### tetrahedra-sdf-grid

| name                                 | type              | description                                                                                                                                                                                                                                                                      |
| ------------------------------------ | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| isosurface_resolution                | int               | Tetrahedra grid resolution for surface extraction. Default: 128                                                                                                                                                                                                                  |
| isosurface_deformable_grid           | bool              | Whether to optimize positions of tetrahedra grid vertices for surface extraction. Default: True                                                                                                                                                                                  |
| isosurface_remove_outliers           | bool              | Whether to remove outlier components according to the number of faces. Only remove if the isosurface process does not require gradient. Default: False                                                                                                                           |
| isosurface_outlier_n_faces_threshold | Union[int, float] | Extracted mesh components with number of faces less than this threshold will be removed if `isosurface_remove_outliers=True`. If `int`, direcly used as the threshold number of faces; if `float`, used as the ratio of all face numbers to compute the threshold. Default: 0.01 |
| geometry_only                        | bool              | Whether to only model the SDF. If True, the feature prediction is ommited. Default:False                                                                                                                                                                                         |
| fix_geometry                         | bool              | Whether to optimize the geometry. If True, the SDF (and grid vertices if `isosurface_deformable_grid=True`) is fixed. Default: False                                                                                                                                             |

### Custom mesh

|  shape_init                   | str        | The shape to initializa the SDF as. Should be formatted as "mesh:path", where `path` points to the custom mesh. Default: ""                            |
| shape_init_params            | Optional[Any]       | Parameters to specify the SDF initialization. A single float is used for uniform scaling; a tuple of three floats is used for scalings along x/y/z axis. Default: None   |

## Material

The material module outputs colors or color latents conditioned on the sampled positions, view directions, and sometimes light directions and normals.

### neural-radiance-material

A material with view dependent effects, parameterized with a network(MLP), similar with that in NeRF.

| name                | type | description                                                                                                                   |
| ------------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------- |
| input_feature_dims  | int  | The dimensions of the input feature. Default: 8                                                                               |
| color_activation    | str  | The activation mapping the network output to the color. Default: "sigmoid"                                                    |
| dir_encoding_config | dict | The config of the positional encoding applied on the ray direction. Default: {"otype": "SphericalHarmonics", "degree": 3}     |
| mlp_network_config  | dict | The config of the MLP network. Default: { "otype": "VanillaMLP", "activation": "ReLU", "n_neurons": 16, "n_hidden_layers": 2} |

### pbr-material

A physically-based rendering (PBR) material.
Currently we support learning albedo, metallic, and roughness. (normal is not supported currently.)

| name                | type  | description                                                                                                      |
| ------------------- | ----- | ---------------------------------------------------------------------------------------------------------------- |
| material_activation | str   | The activation mapping the network output to the materials (albedo, metallic, and roughness). Default: "sigmoid" |
| environment_texture | str   | Path to the environment light map file (`*.hdr`). Default: "load/lights/aerodynamics_workshop_2k.hdr"            |
| environment_scale   | float | Scale of the environment light pixel values. Default: 2.0                                                        |
| min_metallic        | float | Minimum value for metallic. Default: 0.0                                                                         |
| max_metallic        | float | Maximum value for metallic. Default: 0.9                                                                         |
| min_roughness       | float | Minimum value for roughness. Default: 0.08                                                                       |
| max_roughness       | float | Maximum value for roughness. Default: 0.9                                                                        |
| use_bump            | bool  | Whether to train with tangent-space normal perturbation. Default: True                                           |

### no-material

A material without view dependet effects, just map features to colors.

| name               | type           | description                                                                                                                                                                         |
| ------------------ | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| n_output_dims      | int            | The dimensions of the material color, e.g. 3 for RGB and 4 for latent. Default: 3                                                                                                   |
| color_activation   | str            | The activation mapping the network output or the feature to the color. Default: "sigmoid"                                                                                           |
| mlp_network_config | Optional[dict] | The config of the MLP network. Set to `None` to directly map the input feature to the color with `color_activation`, otherwise the feature first goes through an MLP. Default: None |
| input_feature_dims | Optional[int]  | The dimensions of the input feature. Required when use an MLP. Default: None                                                                                                        |

### diffuse-with-point-light-material

| name                | type                     | description                                                                                                                                                                           |
| ------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ambient_light_color | Tuple[float,float,float] | The ambient light color for lambertian shading, used when `soft_shading=False`. Default: (0.1,0.1,0.1)                                                                                |
| diffuse_light_color | Tuple[float,float,float] | The diffuse light color for lambertian shading, used when `soft_shading=False`. Default: (0.9,0.9,0.9)                                                                                |
| ambient_only_steps  | int                      | Number of steps that use albedo color as input to the guidance. Default: 1000                                                                                                         |
| diffuse_prob        | float                    | Use shaded color with a probability of `diffuse_prob` and albedo color with a probability of `1-diffuse_prob` after `ambient_only_steps`. Default: 0.75                               |
| textureless_prob    | float                    | Use textureless shaded color with a probability of `textureless_prob` and lambertian shaded color with a probability of `1-textureless_prob`when using shaded color. Default: 0.5     |
| albedo_activation   | str                      | Activation function for the albedo color. Default: "sigmoid"                                                                                                                          |
| soft_shading        | bool                     | If True, uses a soft version of lambertian shading in training, which randomly samples the ambient light color and diffuse light color. Proposed in the Magic3D paper. Default: False |

### sd-latent-adapter-material

No specific configuration.

## Background

The background should output colors or color latents conditioned on the ray directions.

**Common configurations for background**

| name          | type | description                                                                        |
| ------------- | ---- | ---------------------------------------------------------------------------------- |
| n_output_dims | int  | The dimension of the background color, e.g. 3 for RGB and 4 for latent. Default: 3 |

### solid-color-background

A background with a solid color.

| name    | type  | description                                                                                                              |
| ------- | ----- | ------------------------------------------------------------------------------------------------------------------------ |
| color   | tuple | The initialized color of the background with each value in [0,1], should match `n_output_dims`. Default: (1.0, 1.0, 1.0) |
| learned | bool  | Whether to optimize the background. Default: True                                                                        |

### textured-background

A background with colors parameterized with a texture map.

| name             | type | description                                                                 |
| ---------------- | ---- | --------------------------------------------------------------------------- |
| height           | int  | The height of the texture map. Default: 64                                  |
| width            | int  | The width of the texture map. Default: 64                                   |
| color_activation | str  | The activation mapping the texture feature to the color. Default: "sigmoid" |

### neural-environment-map-background

A background parameterized with a neural network (MLP).

| name                | type                               | description                                                                                                                   |
| ------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| color_activation    | str                                | The activation mapping the network output to the color. Default: "sigmoid"                                                    |
| dir_encoding_config | dict                               | The config of the positional encoding applied on the ray direction. Default: {"otype": "SphericalHarmonics", "degree": 3}     |
| mlp_network_config  | dict                               | The config of the MLP network. Default: { "otype": "VanillaMLP", "activation": "ReLU", "n_neurons": 16, "n_hidden_layers": 2} |
| random_aug          | bool                               | Whether to use random color augmentation. May be able to improve the correctness of the model. Default: False                 |
| random_aug_prob     | float                              | The probability to use random color augmentation. Default: 0.5.                                                               |
| eval_color          | Optional[Tuple[float,float,float]] | The color used in validation/testing. Default: None                                                                           |

## Renderers

Renderers takes geometry, material, and background to produce images given camera and light specifications.

**Common configurations for renderers**

| name   | type  | description                                                                                                                 |
| ------ | ----- | --------------------------------------------------------------------------------------------------------------------------- |
| radius | float | Half side length of the scene bounding box. This should be the same as `radius` of the geometry in most cases. Default: 1.0 |

### nerf-volume-renderer

| name                                                              | type  | description                                                                                                                                                                                                                   |
| ----------------------------------------------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| num_samples_per_ray                                               | float | Number of sample points along each ray. Default: 1.0                                                                                                                                                                          |
| randomized                                                        | bool  | Whether to randomly perturb the sample points in training. Default: True                                                                                                                                                      |
| eval_chunk_size                                                   | int   | Number of sample points per chunk in validation/testing, to prevent OOM. Default: 160000                                                                                                                                      |
| estimator                                                         | str   | The type of sampling estimator. Shoule be one of [occgrid, proposal, importance]. Default: occgrid.                                                                                                                           |
| grid_prune (applicable when using occgrid)                        | bool  | Whether to maintain an occupancy grid and prune sample points in empty space using NeRFAcc. Default: True                                                                                                                     |
| prune_alpha_threshold (applicable when using occgrid)             | bool  | Whether to prune sample points with lower density, only effective when `grid_prune=true`. Default: True                                                                                                                       |
| proposal_network_config (applicable when using proposal)          | dict  | The proposal network configuration, used for density estimation. Default: None                                                                                                                                                |
| prop_optimizer_config (applicable when using proposal)            | dict  | The optimizer configuration for the proposal network. Note that the renderer is not a part of the system's trainable parameters. So the optimizer should be manually specified here, and the optimization is take by Nerfacc. |
| prop_scheduler_config (applicable when using proposal)            | dict  | The learning scheduler for the above optimizer. Default: None                                                                                                                                                                 |
| num_samples_per_ray_proposal (applicable when using proposal)     | int   | Number of sample points along each ray for proposal network. Will sample `num_samples_per_ray` points according to the proposal sampling. Default: 64                                                                         |
| num_samples_per_ray_importance (applicable when using importance) | int   | Number of sample points in NeRF coarse sampling and `num_samples_per_ray` is for fine sampling Default: 64                                                                                                                    |



### neus-volume-renderer

| name                                                              | type  | description                                                                                                |
| ----------------------------------------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------- |
| num_samples_per_ray                                               | float | Number of sample points along each ray. Default: 1.0                                                       |
| randomized                                                        | bool  | Whether to randomly perturb the sample points in training. Default: True                                   |
| eval_chunk_size                                                   | int   | Number of sample points per chunk in validation/testing, to prevent OOM. Default: 160000                   |
| estimator                                                         | str   | The type of sampling estimator. Shoule be one of [occgrid, importance]. Default: occgrid.                  |
| grid_prune (applicable when using occgrid)                        | bool  | Whether to maintain an occupancy grid and prune sample points in empty space using NeRFAcc. Default: True  |
| prune_alpha_threshold (applicable when using occgrid)             | bool  | Whether to prune sample points with lower density, only effective when `grid_prune=true`. Default: True    |
| num_samples_per_ray_importance (applicable when using importance) | int   | Number of sample points in NeRF coarse sampling and `num_samples_per_ray` is for fine sampling Default: 64 |
| learned_variance_init                                             | float | Initialized value for the learned surface variance. Default: 0.3                                           |
| cos_anneal_end_steps                                              | int   | End steps for the linear cosine annealing technique proposed in the NeuS paper. Default: 0                 |
| use_volsdf                                                        | bool  | Whether to use the VolSDF formulation for SDF-to-alpha conversion. Default: False                          |
| near_plane                                                        | float | Distance from camera to the near plane. Default: 0.0                                                       |
| far_plane                                                         | float | Distance from camera to the far plane. Default: 1e10                                                       |

### nvdiff-rasterizer

| name         | type | description                                                                                                                                                                                      |
| ------------ | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| context_type | str  | Rasterization context type used by nvdiffrast, in ["gl", "cuda"]. See the [nvdiffrast documentation](https://nvlabs.github.io/nvdiffrast/#rasterizing-with-cuda-vs-opengl-new) for more details. |

### patch-renderer

The patch-renderer first renders a full low-resolution downsampled image and then randomly renders a local patch at the original resolution level, which can significantly reduce memory usage during high-resolution training.
| name               | type                  | description                                                             |
| ------------------ | --------------------- | ----------------------------------------------------------------------- |
| patch_size         | int                   | The size of the local patch. Default: 128                               |
| global_downsample  | int                   | Downsample scale of the original rendering size. Default: 4             |
| global_detach      | bool                  | Whether to detach the gradient of the downsampled image. Default: False |
| base_renderer_type | str                   | The type of base renderer.                                              |
| base_renderer      | VolumeRenderer.Config | The configuration of the base renderer.                                 |

## Guidance

Given an image or its latent input, the guide should provide its gradient conditioned on a text input so that the image can be optimized with gradient descent to better match the text.

**Common configurations for guidance**

| name                              | type          | description                                                                                                                                                                                                                                                  |
| --------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| enable_memory_efficient_attention | bool          | Whether to enable memory efficient attention in xformers. This will lead to lower GPU memory usage and a potential speed up at inference. Speed up at training time is not guaranteed. Default: false                                                        |
| enable_sequential_cpu_offload     | bool          | Whether to offload all models to CPU. This will use `accelerate`, significantly reducing memory usage but slower. Default: False                                                                                                                             |
| enable_attention_slicing          | bool          | Whether to use sliced attention computation. This will save some memory in exchange for a small speed decrease. Default: False                                                                                                                               |
| enable_channels_last_format       | bool          | Whether to use Channels Last format for the unet. Default: False (Stable Diffusion) / True (DeepFloyd)                                                                                                                                                       |
| pretrained_model_name_or_path     | str           | The pretrained model path in huggingface. Default: "runwayml/stable-diffusion-v1-5" (for `stable-diffusion-guidance`) / "DeepFloyd/IF-I-XL-v1.0" (for `deep-floyd-guidance`) / "stabilityai/stable-diffusion-2-1-base" (for `stable-diffusion-vsd-guidance`) |
| guidance_scale                    | float         | The classifier free guidance scale. Default: 100.0 (for `stable-diffusion-guidance`) / 20.0 (for `deep-floyd-guidance`)                                                                                                                                      |
| grad_clip                         | Optional[Any] | The gradient clip value. None or float or a list in the form of [start_step, start_value, end_value, end_step]. Default: None                                                                                                                                |
| half_precision_weights            | bool          | Whether to use float16 for the diffusion model. Default: True                                                                                                                                                                                                |
| min_step_percent                  | float         | The precent range (min value) of the random timesteps to add noise and denoise. Default: 0.02                                                                                                                                                                |
| max_step_percent                  | float         | The precent range (max value) of the random timesteps to add noise and denoise. Default: 0.98                                                                                                                                                                |
| weighting_strategy                | str           | The choice of w(t) of the sds loss, in ["sds", "uniform", "fantasia3d"]. Default: "sds"                                                                                                                                                                      |
| view_dependent_prompting          | bool          | Whether to use view dependent prompt, i.e. add front/side/back/overhead view to the original prompt. Default: True                                                                                                                                           |

For the first three options, you can check more details in [pipe_stable_diffusion.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py) and [pipeline_if.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py) in diffusers.

### stable-diffusion-guidance

| name                      | type           | description                                                                                                                                         |
| ------------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| use_sjc                   | bool           | Whether to use score jacobian chaining (SJC) instead of SDS. Default: False                                                                         |
| var_red                   | bool           | Whether to use Eq. 16 in [SJC paper](https://arxiv.org/pdf/2212.00774.pdf). Default: True                                                           |
| token_merging             | bool           | Whether to use token merging. This will speed up the unet forward and slightly affect the performance. Default: False                               |
| token_merging_params      | Optional[dict] | The config for token merging. See [here](https://github.com/dbolya/tomesd/blob/main/tomesd/patch.py#L183-L213) for supported arguments. Default: {} |
| anneal_start_step         | Optional[int]  | If specified, denotes at which step to perform t annealing. Default: None                                                                           |

### deep-floyd-guidance

No specific configuration.

## stable-diffusion-vsd-guidance

| name                               | type          | description                                                                                                                                                     |
| ---------------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| pretrained_model_name_or_path_lora | str           | The pretrained base model path for the LoRA model. Default: "stabilityai/stable-diffusion-2-1"                                                                  |
| guidance_scale_lora                | float         | The classifier free guidance scale for the LoRA model. Default: 1.                                                                                              |
| lora_cfg_training                  | bool          | Whether to adopt classifier free guidance training strategy in LoRA training. If True, will zero out the camera condition with a probability 0.1. Default: True |
| camera_condition_type              | str           | Which to use as the camera condition for the LoRA model, in ["extrinsics", "mvp"]. Default: "extrinsics"                                                        |

## Prompt Processors

Prompt processors take a user prompt and compute text embeddings for training. The type of the prompt processor should match that of the guidance.

**Common configurations for prompt processors**

| name                                           | type                     | description                                                                                                                                                                                                                                      |
| ---------------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| prompt                                         | str                      | The text prompt. Default: "a hamburger"                                                                                                                                                                                                          |
| prompt_front                                    | str                     | Manually assigned prompt for the front view. If `None`, use the same as `prompt`. Default: None                                                                                                                                                   |
| prompt_side                                    | str                      | Manually assigned prompt for the side view. If `None`, use the same as `prompt`. Default: None                                                                                                                                                   |
| prompt_back                                    | str                      | Manually assigned prompt for the back view. If `None`, use the same as `prompt`. Default: None                                                                                                                                                   |
| prompt_overhead                                | str                      | Manually assigned prompt for the overhead view. If `None`, use the same as `prompt`. Default: None                                                                                                                                               |
| negative_prompt                                | str                      | The uncondition text input in Classifier Free Guidance. Default: ""                                                                                                                                                                              |
| pretrained_model_name_or_path                  | str                      | The pretrained model path in huggingface. Default: "runwayml/stable-diffusion-v1-5" (for `stable-diffusion-prompt-processor`) / "DeepFloyd/IF-I-XL-v1.0" (fpr `deep-floyd-prompt-processor`)                                                     |
| overhead_threshold                             | float                    | Consider the view as overhead when the elevation degree > overhead_threshold. Default: 60.0                                                                                                                                                      |
| front_threshold                                | float                    | Consider the view as front when the azimuth degree in [-front_threshold, front_threshold]. Default: 45.0                                                                                                                                         |
| back_threshold                                 | float                    | Consider the view as back when the azimuth degree > 180 - back_threshold or < -180 + back_threshold. Default: 45.0                                                                                                                               |
| view_dependent_prompt_front                    | bool                     | Whether to put the vide dependent prompt in front of the original prompt. If set to True, the final prompt will be `a front/back/side/overhead view of [prompt]`, otherwise it will be `[prompt], front/back/side/overhead view`. Default: False |
| use_cache                                      | bool                     | Whether to cache computed text embeddings. If True, will use cached text embeddings if available. Default: True                                                                                                                                  |
| spawn                                          | bool                     | Whether to spawn a new process to compute text embeddings. Must set to True if using multiple GPUs and DeepFloyd-IF guidance. Default: True                                                                                                      |
| use_perp_neg                                   | bool                     | Whether to use the Perp-Neg algorithm to alleviate the multi-face problem. Default: False                                                                                                                                                        |
| perp_neg_f_sb                                  | Tuple[float,float,float] |                                                                                                                                                                                                                                                  |
| perp_neg_f_fsb                                 | Tuple[float,float,float] |                                                                                                                                                                                                                                                  |
| perp_neg_f_fs                                  | Tuple[float,float,float] |                                                                                                                                                                                                                                                  |
| perp_neg_f_sf                                  | Tuple[float,float,float] |                                                                                                                                                                                                                                                  |
| use_prompt_debiasing                           | bool                     | Whether to use the prompt debiasing algorithm to compute debiased view-dependent prompts. Default: False                                                                                                                                         |
| pretrained_model_name_or_path_prompt_debiasing | str                      | The pretrained model path for prompt debiasing. Default: "bert-base-uncased"                                                                                                                                                                     |
| prompt_debiasing_mask_ids                      | Optional[List[int]]      | Index of words that can potentially be removed in prompt debiasing. If `None`, all words can be removed. Default: None                                                                                                                           |

### stable-diffusion-prompt-processor

No specific configuration.

### deep-floyd-prompt-processor

No specific configuration.

## Exporters

Exporters output assets like textured meshes, which can be used for further processing.

**Common configurations for exporters**

| name       | type | description                                 |
| ---------- | ---- | ------------------------------------------- |
| save_video | bool | Whether to save a 360 video. default: False |

### mesh-exporter

| name                 | type | description                                                                                                                                                                                                                                                       |
| -------------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| fmt                  | str  | The format to save, in ["obj-mtl", "obj"]. If "obj-mtl", save to an obj file with mtl material specification; if "obj", save to an obj file with vertex colors. Default: "obj-mtl"                                                                                |
| save_name            | str  | Filename of the saved mesh model, without extension. Default: "model"                                                                                                                                                                                             |
| save_normal          | bool | Whether to save vertex normal. Default: False                                                                                                                                                                                                                     |
| save_uv              | bool | Whether to save texture coordinates. If True, will use xatlas to perform UV unwrapping. Default: True                                                                                                                                                             |
| save_texture         | bool | Whether to save texture information. If True, will save texture maps if `fmt="obj-mtl"`, and will save vertex colors if `fmt="obj"`. Note that `save_uv` must be True for `save_texture=True` and `fmt="obj-mtl"`. Default: True                                  |
| texture_size         | int  | Texture map size, used when `save_texture=True` and `fmt="obj-mtl"`. Default: 1024                                                                                                                                                                                |
| texture_format       | str  | Texture map file format, used when `save_texture=True` and `fmt="obj-mtl"`. Default: "jpg"                                                                                                                                                                        |
| xatlas_chart_options | dict | Chart options for xatlas UV unwrapping, used when `save_uv=True`. See [here](https://github.com/MozillaReality/xatlas-web/blob/master/xatlas.h#L169) for supported options. Default: {}                                                                           |
| xatlas_pack_options  | dict | Pack options for xatlas UV unwrapping, used when `save_uv=True`. See [here](https://github.com/MozillaReality/xatlas-web/blob/master/xatlas.h#L201) for supported options. Default: {}                                                                            |
| context_type         | str  | Rasterization context type used by nvdiffrast, in ["gl", "cuda"], used when `save_texture=True` and `fmt="obj-mtl"`. See the [nvdiffrast documentation](https://nvlabs.github.io/nvdiffrast/#rasterizing-with-cuda-vs-opengl-new) for more details. Default: "gl" |

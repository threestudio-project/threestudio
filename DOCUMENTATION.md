## Common Configurations

| name          | type          | description                                                                                                                                                                                                                                                           |
| ------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| name          | str           | Name of the experiment. Default: "default"                                                                                                                                                                                                                            |
| description   | str           | Descrption of the experiment. Default: ""                                                                                                                                                                                                                             |
| tag           | str           | Tag of the experiment. Default: ""                                                                                                                                                                                                                                    |
| seed          | str           | Global seed of the experiment. Used by `seed_everything` of PyTorch-Lightning. Default: 0                                                                                                                                                                             |
| use_timestamp | bool          | Whether to use the current timestamp as the suffix of the tag. Default: True                                                                                                                                                                                          |
| timestamp     | Optional[str] | The timestamp as the suffix of the tag. DO NOT set this manually. Default: None                                                                                                                                                                                       |
| exp_root_dir  | str           | The root directory for outputs of all the experiments. Default: "outputs"                                                                                                                                                                                             |
| exp_dir       | str           | The directory for outputs of the current experiment. DO NOT set this manually. It will be automatically set to `[exp_root_dir]/[name]`.                                                                                                                               |
| trial_name    | str           | Name of the trial. DO NOT set this manually. It will be automatically set to `[tag]@[timestamp]`.                                                                                                                                                                     |
| trial_dir     | str           | The directory for outputs for the current trial. DO NOT set this manually. It will be automatically set to `[exp_root_dir]/[name]/[trial_name].`                                                                                                                      |
| resume        | Optional[str] | The path to the checkpoint file to resume from. Default: None                                                                                                                                                                                                         |
| data_type     | str           | Type of the data module used. See []() for supported data modules. Default: ""                                                                                                                                                                                        |
| data          | dict          | Configurations of the data module. Default: {}                                                                                                                                                                                                                        |
| system_type   | str           | Type of the system used. See []() for supported systems. Default: ""                                                                                                                                                                                                  |
| system        | dict          | Configurations of the system. Defaut: {}                                                                                                                                                                                                                              |
| trainer       | dict          | Configurations of PyTorch-Lightning Trainer. See https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api for supported arguments. Exceptions: `logger` and `callbacks` are set in `launch.py`. Default: {}                                     |
| checkpoint    | dict          | Configurations of PyTorch-Lightning ModelCheckpoint callback, which defines when the checkpoint will be saved. See https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#modelcheckpoint for supported arguments. Default: {} |

## Data

### random-camera-datamodule

| name                   | type               | description                                                                                                                                                                                                                                                                         |
| ---------------------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| height                 | int                | Height of the rendered image in training. Default: 64                                                                                                                                                                                                                               |
| width                  | int                | Width of the rendered image in training. Default: 64                                                                                                                                                                                                                                |
| eval_height            | int                | Height of the rendered image in validation/testing. Default: 512                                                                                                                                                                                                                    |
| eval_width             | int                | Width of the rendered image in validation/testing. Default: 512                                                                                                                                                                                                                     |
| batch_size             | int                | Number of images per batch in training. Default: 1                                                                                                                                                                                                                                  |
| eval_batch_size        | int                | Number of images per batch in validation/testing. DO NOT change this. Default: 1                                                                                                                                                                                                    |
| elevation_range        | Tuple[float,float] | Camera elevation angle range to sample from in training, in degrees. Default: (-10,90)                                                                                                                                                                                              |
| azimuth_range          | Tuple[float,float] | Camera azimuth angle range to sample from in training, in degrees. Default: (-180,180)                                                                                                                                                                                              |
| camera_distance_range  | Tuple[float,float] | Camera distance range to sample from in training. Default: (1,1.5)                                                                                                                                                                                                                  |
| fovy_range             | Tuple[float,float] | Camera field of view (FoV) range along the y direction (vertical direction) to sample from in training, in degrees. Default: (40,70)                                                                                                                                                |
| camera_perturb         | float              | Random perturbation ratio for the sampled camera positions in training. The sampled camera positions will be perturbed by `N(0,1) * camera_perturb`. Default: 0.1                                                                                                                   |
| center_perturb         | float              | Random perturbation ratio for the look-at point of the cameras in training. The look-at point wil be `N(0,1) * center_perturb`. Default: 0.2                                                                                                                                        |
| up_perturb             | float              | Random pertubation ratio for the up direction of the cameras in training. The up direction will be `[0,0,1] + N(0,1) * up_perturb`. Default: 0.02                                                                                                                                   |
| light_position_perturb | float              | Used to get random light directions from camera positions, only used when `light_sample_strategy="dreamfusion"`. The camera positions will be perturbed by `N(0,1) * light_position_perturb`, then the perturbed positions are used to determine the light directions. Default: 1.0 |
| light_distance_range   | Tuple[float,float] | Point light distance range to sample from in training. Default: (0.8,1.5)                                                                                                                                                                                                           |
| eval_elevation_deg     | float              | Camera elevation angle in validation/testing, in degrees. Default: 150                                                                                                                                                                                                              |
| eval_camera_distance   | float              | Camera distance in validation/testing. Default: 15                                                                                                                                                                                                                                  |
| eval_fovy_deg          | float              | Camera field of view (FoV) along the y direction (vertical direction) in validation/testing, in degrees. Default: 70                                                                                                                                                                |
| light_sample_strategy  | str                | Strategy to sample point light positions in training, in ["dreamfusion", "magic3d"]. "dreamfusion" uses strategy described in the DreamFusion paper; "magic3d" uses strategy decribed in the Magic3D paper. Default: "dreamfusion"                                                  |
| batch_uniform_azimuth  | bool               | Whether to ensure the uniformity of sampled azimuth angles in training as described in the Fantasia3D paper. If True, the `azimuth_range` is equally divided into `batch_size` bins and the azimuth angles are sampled from every bins. Default: True                               |

### single-image-datamodule

| name | type | description |
| ---- | ---- | ----------- |

### co3d-datamodule

| name | type | description |
| ---- | ---- | ----------- |

## Systems

Common configurations:

| name                          | type                | description |
| ----------------------------- | ------------------- | ----------- |
| loss                          | dict                |             |
| optimizer                     | dict                |             |
| scheduler                     | Optional[dict]      |             |
| weights                       | Optional[str]       |             |
| weights_ignore_modules        | Optional[List[str]] |             |
| cleanup_after_validation_step | bool                |             |
| cleanup_after_validation_step | bool                |             |

### dreamfusion-system

| name                  | type | description |
| --------------------- | ---- | ----------- |
| geometry_type         | str  |             |
| geometry              | dict |             |
| material_type         | str  |             |
| matrial               | dict |             |
| background_type       | str  |             |
| background            | dict |             |
| renderer_type         | str  |             |
| renderer              | dict |             |
| guidance_type         | str  |             |
| guidance              | dict |             |
| prompt_processor_type | str  |             |
| prompt_processor      | dict |             |

### magic3d-system

### sjc-system

### latentnerf-system

### fantasia3d-system

### image-condition-dreamfusion-system

## Geometry

Common configurations for implicit geometry:

| name | type | description |
| ---- | ---- | ----------- |

### implicit-volume

| name | type | description |
| ---- | ---- | ----------- |

### implicit-sdf

| name | type | description |
| ---- | ---- | ----------- |

### volume-grid

| name | type | description |
| ---- | ---- | ----------- |

Common configurations for explicit geometry:

| name | type | description |
| ---- | ---- | ----------- |

### tetrahedra-sdf-grid

| name | type | description |
| ---- | ---- | ----------- |

## Material

### neural-radiance-material

### no-material

### diffuse-with-point-light-material

### sd-latent-adapter-material

## Background

### solid-color-background

### textured-background

### neural-environment-map-background

## Renderer

### nerf-volume-renderer

### nvdiff-rasterizer

## Guidance

### stable-diffusion-guidance

### deep-floyd-guidance

## Prompt Processor

### dreamfusion-prompt-processor

### deep-floyd-prompt-processor

## Common Configurations

| name          | type          | description                                                                                                                             |
| ------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| name          | str           | Name of the experiment. Default: "default"                                                                                              |
| description   | str           | Descrption of the experiment. Default: ""                                                                                               |
| tag           | str           | Tag of the experiment. Default: ""                                                                                                      |
| seed          | str           | Global seed of the experiment. Used by `seed_everything` of PyTorch Lightning. Default: 0                                               |
| use_timestamp | bool          | Whether to use the current timestamp as the suffix of the tag. Default: True                                                            |
| timestamp     | Optional[str] | The timestamp as the suffix of the tag. DO NOT set this manually. Default: None                                                         |
| exp_root_dir  | str           | The root directory for outputs of all the experiments. Default: "outputs"                                                               |
| exp_dir       | str           | The directory for outputs of the current experiment. DO NOT set this manually. It will be automatically set to `[exp_root_dir]/[name]`. |
| trial_name    | str           | Name of the trial. DO NOT set this manually. It will be automatically set to `[tag]@[timestamp]`.                                       |
| trial_dir     | str           |                                                                                                                                         |
| resume        | Optional[str] |                                                                                                                                         |
| data_type     | str           |                                                                                                                                         |
| data          | dict          |                                                                                                                                         |
| system_type   | str           |                                                                                                                                         |
| system        | dict          |                                                                                                                                         |
| trainer       | dict          |                                                                                                                                         |
| checkpoint    | dict          |                                                                                                                                         |

## Data

### random-camera-datamodule

| name   | type | description |
| ------ | ---- | ----------- |
| height | int  |             |
| width  | int  |             |
|        |      |             |

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

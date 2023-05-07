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

The background should output colors or color latents conditioned on the ray directions.

### solid-color-background

A background with a solid color.

| name | type | description |
| ---- | ---- | ----------- |
| n_output_dims | int | The color dimension of the background,e.g. 3 for RGB and 4 for latent. Default: 3 |
| color| tuple| The initialized color of the background with each value in [0,1], should match `n_output_dims`. Default: (1.0, 1.0, 1.0) |
| learned | bool | Whether to optimize the background. Default: True |

### textured-background

A background with colors parameterized with a texture map.

| name | type | description |
| ---- | ---- | ----------- |
| n_output_dims | int | The color dimension of the background,e.g. 3 for RGB and 4 for latent. Default: 3 |
| height | int | The height of the texture map. Default: 64 |
| width | int | The width of the texture map. Default: 64 |
| color_activation | str | The activation applied on the texture map. Default: "sigmoid" |

### neural-environment-map-background

A background parameterized with a neural network (MLP).

| name | type | description |
| ---- | ---- | ----------- |
| n_output_dims | int | The color dimension of the background,e.g. 3 for RGB and 4 for latent. Default: 3 |
| color_activation | str | The activation applied on the network output. Default: "sigmoid" |
| dir_encoding_config | dict | The config of the positional encoding applied on the ray direction. Default: {"otype": "SphericalHarmonics", "degree": 3} |
| mlp_network_config | dict | The config of the MLP network. Default: { "otype": "VanillaMLP", "activation": "ReLU", "n_neurons": 16, "n_hidden_layers": 2} |
| random_aug | bool | Whether to use random color augmentation. May be able to improve the correctness of the model. Default: False |
| random_aug_prob | float | The probability to use random color augmentation. Default: 0.5. |

## Renderer

### nerf-volume-renderer

### nvdiff-rasterizer

## Guidance

### stable-diffusion-guidance

### deep-floyd-guidance

## Prompt Processor

### dreamfusion-prompt-processor

### deep-floyd-prompt-processor

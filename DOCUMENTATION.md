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

The material should output colors or color latents conditioned on the sampled positions, view directions, and sometimes light directions and normals.

### neural-radiance-material

A material with view dependent effects, parameterized with a network(MLP), similar with that in NeRF.

| name | type | description |
| ---- | ---- | ----------- |
| input_feature_dims | int | The dimensions of the input feature. Default: 8 |
| color_activation | str | The activation mapping the network output to the color. Default: "sigmoid" |
| dir_encoding_config | dict | The config of the positional encoding applied on the ray direction. Default: {"otype": "SphericalHarmonics", "degree": 3} |
| mlp_network_config | dict | The config of the MLP network. Default: { "otype": "VanillaMLP", "activation": "ReLU", "n_neurons": 16, "n_hidden_layers": 2} |

### no-material

A material without view dependet effects, just map features to colors.

| name | type | description |
| ---- | ---- | ----------- |
| n_output_dims | int | The dimensions of the material color, e.g. 3 for RGB and 4 for latent. Default: 3 |
| color_activation | str | The activation mapping the network output or the feature to the color. Default: "sigmoid" |
| mlp_network_config | Optional[dict] | The config of the MLP network. Set to `None` to directly map the input feature to the color with `color_activation`, otherwise the feature first goes through an MLP. Default: None |
| input_feature_dims | Optional[int] | The dimensions of the input feature. Required when use an MLP. Default: None |

### diffuse-with-point-light-material

| name | type | description |
| ---- | ---- | ----------- |

### sd-latent-adapter-material

| name | type | description |
| ---- | ---- | ----------- |

## Background

The background should output colors or color latents conditioned on the ray directions.

**Common configurations for background**

| name | type | description |
| ---- | ---- | ----------- |
| n_output_dims | int | The dimension of the background color, e.g. 3 for RGB and 4 for latent. Default: 3 |

### solid-color-background

A background with a solid color.

| name | type | description |
| ---- | ---- | ----------- |
| color| tuple| The initialized color of the background with each value in [0,1], should match `n_output_dims`. Default: (1.0, 1.0, 1.0) |
| learned | bool | Whether to optimize the background. Default: True |

### textured-background

A background with colors parameterized with a texture map.

| name | type | description |
| ---- | ---- | ----------- |
| height | int | The height of the texture map. Default: 64 |
| width | int | The width of the texture map. Default: 64 |
| color_activation | str | The activation mapping the texture feature to the color. Default: "sigmoid" |

### neural-environment-map-background

A background parameterized with a neural network (MLP).

| name | type | description |
| ---- | ---- | ----------- |
| color_activation | str | The activation mapping the network output to the color. Default: "sigmoid" |
| dir_encoding_config | dict | The config of the positional encoding applied on the ray direction. Default: {"otype": "SphericalHarmonics", "degree": 3} |
| mlp_network_config | dict | The config of the MLP network. Default: { "otype": "VanillaMLP", "activation": "ReLU", "n_neurons": 16, "n_hidden_layers": 2} |
| random_aug | bool | Whether to use random color augmentation. May be able to improve the correctness of the model. Default: False |
| random_aug_prob | float | The probability to use random color augmentation. Default: 0.5. |

## Renderer

### nerf-volume-renderer

### nvdiff-rasterizer

## Guidance

Given an image or its latent input, the guide should provide its gradient conditioned on a text input so that the image can be optimized with gradient descent to better match the text.

**Common configurations for guidance**

| name | type | description |
| ---- | ---- | ----------- |
| enable_memory_efficient_attention | bool | Whether to enable memory efficient attention in xformers. This will lead to lower GPU memory usage and a potential speed up at inference. Speed up at training time is not guaranteed. Default: false |
| enable_sequential_cpu_offload | bool | Whether to offload all models to CPU. This will use `accelerate`, significantly reducing memory usage but slower. Default: False |
| enable_attention_slicing | bool | Whether to use sliced attention computation. This will save some memory in exchange for a small speed decrease. Default: False |
| enable_channels_last_format | bool | Whether to use Channels Last format for the unet. Default: False (Stable Diffusion) / True (DeepFloyd) |
| pretrained_model_name_or_path | str | The pretrained model path in huggingface. Default: "runwayml/stable-diffusion-v1-5" (Stable Diffusion) / "DeepFloyd/IF-I-XL-v1.0" (DeepFloyd) |
| guidance_scale | float | The classifier free guidance scale. Default: 100.0 (Stable Diffusion) / 20.0 (DeepFloyd) |
| grad_clip | Optional[Any] | The gradient clip value. None or float or a list in the form of [start_step, start_value, end_value, end_step]. Default: None |
| half_precision_weights | bool | Whether to use float16 for the diffusion model. Default: True |
| min_step_percent | float | The precent range (min value) of the random timesteps to add noise and denoise. Default: 0.02 |
| max_step_percent | float | The precent range (max value) of the random timesteps to add noise and denoise. Default: 0.98 |
| weighting_strategy | str | The choice of w(t) of the sds loss. Default: "sds" |

For the first three options, you can check more details in  [pipe_stable_diffusion.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py) and [pipeline_if.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py) in diffusers.

### stable-diffusion-guidance
| name | type | description |
| ---- | ---- | ----------- |
| use_sjc | bool | Whether to use score jacobian chaining (SJC) instead of SDS. Default: False |
| var_red | bool | Whether to use Eq. 16 in [SJC paper](https://arxiv.org/pdf/2212.00774.pdf). Default: True |
| token_merging | bool | Whether to use token merging. This will speed up the unet forward and slightly affect the performance. Default: False |
| token_merging_params | Optional[dict] | The config for token merging. Default: {} |

### deep-floyd-guidance

No specific configuration.

## Prompt Processor

**Common configurations for prompt processor**

| name | type | description |
| ---- | ---- | ----------- |
| prompt | str | The text prompt. Default: "a hamburger" |
| negative_prompt | str | The uncondition text input in Classifier Free Guidance. Default: "" |
| pretrained_model_name_or_path | str | The pretrained model path in huggingface. Default: "runwayml/stable-diffusion-v1-5" (Stable Diffusion) / "DeepFloyd/IF-I-XL-v1.0" (DeepFloyd) |
| view_dependent_prompting | bool | Whether to use view dependent prompt, i.e. add front/side/back/overhead view to the original prompt. Default: True |
| overhead_threshold | float | Consider the view as overhead when the elevation degree > overhead_threshold. Default: 60.0 |
| front_threshold | float | Consider the view as front when the azimuth degree in [-front_threshold, front_threshold]. Default: 45.0 |
| back_threshold | float | Consider the view as back when the azimuth degree > 180 - back_threshold or < -180 + back_threshold. Default: 45.0 |
| view_dependent_prompt_front | bool | Whether to put the vide dependent prompt in front of the original prompt. If set to True, the final prompt will be `a front/back/side/overhead view of [prompt]`, otherwise it will be `[prompt], front/back/side/overhead view`. Default: False |

### dreamfusion-prompt-processor

No specific configuration.

### deep-floyd-prompt-processor

No specific configuration.

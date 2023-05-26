<p align="center">
    <picture>
    <img alt="threestudio" src="https://user-images.githubusercontent.com/19284678/236847132-219999d0-4ffa-4240-a262-c2c025d15d9e.png" width="50%">
    </picture>
</p>

<p align="center"><b>
threestudio is a unified framework for 3D content creation from text prompts, single images, and few-shot images, by lifting 2D text-to-image generation models.
</b></p>

<p align="center">
<img alt="threestudio" src="https://user-images.githubusercontent.com/3117031/236739017-365626d9-bb35-4c47-b71d-b9de767b0644.gif" width="100%">
</p>

<p align="center"><b>
👆 Results obtained from methods implemented by threestudio 👆 <br/>
| <a href="https://dreamfusion3d.github.io/">DreamFusion</a> | <a href="https://research.nvidia.com/labs/dir/magic3d/">Magic3D</a> | <a href="https://pals.ttic.edu/p/score-jacobian-chaining">SJC</a> | <a href="https://github.com/eladrich/latent-nerf">Latent-NeRF</a> | <a href="https://fantasia3d.github.io/">Fantasia3D</a> |
</b></p>

<p align="center">
  <a href="https://colab.research.google.com/github/threestudio-project/threestudio/blob/main/threestudio.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg">
  </a>
</p>

<p align="center">
    Did not find what you want? Submit a feature request or upvote others' requests <a href="https://github.com/threestudio-project/threestudio/discussions/46">here</a>!
</p>

## News

- 05/26/2023: An experimental implementation of ProlificDreamer! Following the instruction [here](https://github.com/threestudio-project/threestudio#prolificdreamer-) to have a try.
- 05/14/2023: You can experiment with the SDS loss on 2D images using our [2dplayground](2dplayground.ipynb).
- 05/13/2023: You can now try threestudio on [Google Colab](https://colab.research.google.com/github/threestudio-project/threestudio/blob/main/threestudio.ipynb)!
- 05/11/2023: We now support exporting textured meshes! See [here](https://github.com/threestudio-project/threestudio#export-meshes) for instructions.

![export-blender](https://github.com/threestudio-project/threestudio/assets/19284678/ccae2820-e702-484c-a43f-81678a365427)

## Installation

The following steps have been tested on Ubuntu20.04.

- You must have a NVIDIA graphics card with at least 6GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
- Install `Python >= 3.8`.
- (Optional, Recommended) Create a virtual environment:

```sh
python3 -m virtualenv venv
. venv/bin/activate
```

- Install `PyTorch >= 1.12`. We have tested on `torch1.12.1+cu113` and `torch2.0.0+cu118`, but other versions should also work fine.

```sh
# torch1.12.1+cu113
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# or torch2.0.0+cu118
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

- (Optional, Recommended) Install ninja to speed up the compilation of CUDA extensions:

```sh
pip install ninja
```

- Install dependencies:

```sh
pip install -r requirements.txt
```

- (Optional, Recommended) The best-performing models in threestudio uses the newly-released T2I model [DeepFloyd IF](https://github.com/deep-floyd/IF) which currently requires signing a license agreement. If you would like use these models, you need to [accept the license on the model card of DeepFloyd IF](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0), and login in the Hugging Face hub in terminal by `huggingface-cli login`.

- For contributors, see [here](https://github.com/threestudio-project/threestudio#contributing-to-threestudio).

## Quickstart

Here we show some basic usage of threestudio. First let's train a DreamFusion model to create a classic pancake bunny.

**If you are experiencing unstable connections with Hugging Face, we suggest you either (1) setting environment variable `TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1` before your running command after all needed files have been fetched on the first run, to prevent from connecting to Hugging Face each time you run, or (2) downloading the guidance model you used to a local folder following [here](https://huggingface.co/docs/huggingface_hub/v0.14.1/guides/download#download-an-entire-repository) and [here](https://huggingface.co/docs/huggingface_hub/v0.14.1/guides/download#download-files-to-local-folder), and set `pretrained_model_name_or_path` of the guidance and the prompt processor to the local path.**

```sh
# if you have agreed the license of DeepFloyd IF and have >20GB VRAM
# please try this configuration for higher quality
python launch.py --config configs/dreamfusion-if.yaml --train --gpu 0 system.prompt_processor.prompt="a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
# otherwise you could try with the Stable Diffusion model, which fits in 6GB VRAM
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes"
```

threestudio uses [OmegaConf](https://github.com/omry/omegaconf) for flexible configurations. You can easily change any configuration in the YAML file by specifying arguments without `--`, for example the specified prompt in the above cases. For all supported configurations, please see our [documentation](https://github.com/threestudio-project/threestudio/blob/main/DOCUMENTATION.md).

The training lasts for 10,000 iterations. You can find visualizations of the current status in the trial directory which defaults to `[exp_root_dir]/[name]/[tag]@[timestamp]`, where `exp_root_dir` (`outputs/` by default), `name` and `tag` can be set in the configuration file. A 360-degree video will be generated after the training is completed. In training, press `ctrl+c` one time will stop training and head directly to the test stage which generates the video. Press `ctrl+c` the second time to fully quit the program.

### Multi-GPU training

Multi-GPU training is supported. Note that `data.batch_size` is the batch size **per rank (device)**. Also remember to

- Set `data.n_val_views` to be a multiple of the number of GPUs.
- Set a unique `tag` as timestamp is disabled in multi-GPU training and will not be appended after the tag. If you the same tag as previous trials, saved config files, code and visualizations will be overriden.

```sh
# this results in an effective batch size of 4 (number of GPUs) * 2 (data.batch_size) = 8
python launch.py --config configs/dreamfusion-if.yaml --train --gpu 0,1,2,3 system.prompt_processor.prompt="a zoomed out DSLR photo of a baby bunny sitting on top of a stack of pancakes" data.batch_size=2 data.n_val_views=4
```

### Resume from checkpoints

If you want to resume from a checkpoint, do:

```sh
# resume training from the last checkpoint, you may replace last.ckpt with any other checkpoints
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 resume=path/to/trial/configs/last.ckpt
# if the training has completed, you can still continue training for a longer time by setting trainer.max_steps
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 resume=path/to/trial/configs/last.ckpt trainer.max_steps=20000
# you can also perform testing using resumed checkpoints
python launch.py --config path/to/trial/dir/configs/parsed.yaml --test --gpu 0 resume=path/to/trial/configs/last.ckpt
# note that the above commands use parsed configuration files from previous trials
# which will continue using the same trial directory
# if you want to save to a new trial directory, replace parsed.yaml with raw.yaml in the command
```

### Export Meshes

To export the scene to texture meshes, use the `--export` option. We currently support exporting to obj+mtl, or obj with vertex colors.

```sh
# this uses default mesh-exporter configurations which exports obj+mtl
python launch.py --config path/to/trial/dir/configs/parsed.yaml --export --gpu 0 resume=path/to/trial/configs/last.ckpt system.exporter_type=mesh-exporter
# specify system.exporter.fmt=obj to get obj with vertex colors
# you may also add system.exporter.save_uv=false to accelerate the process, suitable for a quick peek of the result
python launch.py --config path/to/trial/dir/configs/parsed.yaml --export --gpu 0 resume=path/to/trial/configs/last.ckpt system.exporter_type=mesh-exporter system.exporter.fmt=obj
# for NeRF-based methods (DreamFusion, Magic3D coarse, Latent-NeRF, SJC)
# you may need to adjust the isosurface threshold (25 by default) to get satisfying outputs
# decrease the threshold if the extracted model is incomplete, increase if it is extruded
python launch.py --config path/to/trial/dir/configs/parsed.yaml --export --gpu 0 resume=path/to/trial/configs/last.ckpt system.exporter_type=mesh-exporter system.geometry.isosurface_threshold=10.
# use marching cubes of higher resolutions to get more detailed models
python launch.py --config configs/magic3d-refine-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger" system.from_coarse=path/to/coarse/stage/trial/ckpts/last.ckpt system.geometry.isosurface_method=mc-cpu system.geometry.isosurface_resolution=256
```

For all the options you can specify when exporting, see [the documentation](https://github.com/threestudio-project/threestudio/blob/main/DOCUMENTATION.md#exporters).

See [here](https://github.com/threestudio-project/threestudio#supported-models) for example running commands of all our supported models. Please refer to [here](https://github.com/threestudio-project/threestudio#tips-on-improving-quality) for tips on getting higher-quality results, and [here](https://github.com/threestudio-project/threestudio#vram-optimization) for reducing VRAM usage.

For feature requests, bug reports, or discussions about technical problems, please [file an issue](https://github.com/threestudio-project/threestudio/issues/new). In case you want to discuss the generation quality or showcase your generation results, please feel free to participate in the [discussion panel](https://github.com/threestudio-project/threestudio/discussions).

## Supported Models

### DreamFusion [![arXiv](https://img.shields.io/badge/arXiv-2209.14988-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2209.14988)

**Results obtained by threestudio (DeepFloyd IF, batch size 8)**

https://user-images.githubusercontent.com/19284678/236694848-38ae4ea4-554b-4c9d-b4c7-fba5bee3acb3.mp4

**Notable differences from the paper**

- We use open-source T2I models (StableDiffusion, DeepFloyd IF), while the paper uses Imagen.
- We use a guiandance scale of 20 for DeepFloyd IF, while the paper uses 100 for Imagen.
- We do not use sigmoid to normalize the albedo color but simply scale the color from `[-1,1]` to `[0,1]`, as we find this help convergence.
- We use HashGrid encoding and uniformly sample points along rays, while the paper uses Integrated Positional Encoding and sampling strategy from MipNeRF360.
- We adopt camera settings and density initialization strategy from Magic3D, which is slightly different from the DreamFusion paper.
- Some hyperparameters are different, such as the weighting of loss terms.

**Example running commands**

```sh
# uses DeepFloyd IF, requires ~15GB VRAM to extract text embeddings and ~10GB VRAM in training
# here we adopt random background augmentation to improve geometry quality
python launch.py --config configs/dreamfusion-if.yaml --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger" system.background.random_aug=true
# uses StableDiffusion, requires ~6GB VRAM in training
python launch.py --config configs/dreamfusion-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"
```

**Tips**

- DeepFloyd IF performs **way better than** StableDiffusion.
- Validation shows albedo color before `system.material.ambient_only_steps` and shaded color after that.
- Try increasing/decreasing `system.loss.lambda_sparsity` if your scene is stuffed with floaters/becoming empty.
- Try increasing/decreasing `system.loss.lambda_orient` if you object is foggy/over-smoothed.
- Try replacing the background to random colors with a probability 0.5 by setting `system.background.random_aug=true` if you find the model incorrectly treats the background as part of the object.
- DeepFloyd IF uses T5-XXL as its text encoder, which consumes ~15GB VRAM even when using 8-bit quantization. This is currently the bottleneck for training with less VRAM. If anyone knows how to run the text encoder with less VRAM, please file an issue. We're also trying to push the text encoder to [Replicate](https://replicate.com/) to enable extracting text embeddings via API, but are having some network connection issues. Please [contact bennyguo](mailto:imbennyguo@gmail.com) if you would like to help out.

### Magic3D [![arXiv](https://img.shields.io/badge/arXiv-2211.10440-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2211.10440)

**Results obtained by threestudio (DeepFloyd IF, batch size 8; first row: coarse, second row: refine)**

https://user-images.githubusercontent.com/19284678/236694858-0ed6939e-cd7a-408f-a94b-406709ae90c0.mp4

**Notable differences from the paper**

- We use open-source T2I models (StableDiffusion, DeepFloyd IF) for the coarse stage, while the paper uses eDiff-I.
- In the coarse stage, we use a guiandance scale of 20 for DeepFloyd IF, while the paper uses 100 for eDiff-I.
- In the coarse stage, we use analytic normal, while the paper uses predicted normal.
- In the coarse stage, we use orientation loss as in DreamFusion, while the paper does not.
- There are many things that are ommited from the paper such as the weighting of loss terms and the DMTet grid resolution, which could be different.

**Example running commands**

First train the coarse stage NeRF:

```sh
# uses DeepFloyd IF, requires ~15GB VRAM to extract text embeddings and ~10GB VRAM in training
python launch.py --config configs/magic3d-coarse-if.yaml --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"
# uses StableDiffusion, requires ~6GB VRAM in training
python launch.py --config configs/magic3d-coarse-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"
```

Then convert the NeRF from the coarse stage to DMTet and train with differentiable rasterization:

```sh
# the refinement stage uses StableDiffusion, requires ~5GB VRAM in training
# NOTE: the meaning of system.from_coarse has changed from cfff05, it is now the path to the coarse stage weights instead of a boolean value
python launch.py --config configs/magic3d-refine-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger" system.from_coarse=path/to/coarse/stage/trial/ckpts/last.ckpt
# if you're unsatisfied with the surface extraced using the default threshold (25)
# you can specify a threshold value using `system.coarse_geometry_override`
# decrease the value if the extracted surface is incomplete, increate if it is extruded
python launch.py --config configs/magic3d-refine-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger" system.from_coarse=path/to/coarse/stage/trial/ckpts/last.ckpt system.coarse_geometry_override.isosurface_threshold=10.
```

**Tips**

- For the coarse stage, DeepFloyd IF performs **way better than** StableDiffusion.
- Magic3D uses a neural network to predict the surface normal, which may not resemble the true geometric normal and degrade geometry quality, so we use analytic normal instead.
- Try increasing/decreasing `system.loss.lambda_sparsity` if your scene is stuffed with floaters/becoming empty.
- Try increasing/decreasing `system.loss.lambda_orient` if you object is foggy/over-smoothed.
- Try replacing the background to random colors with a probability 0.5 by setting `system.background.random_aug=true` if you find the model incorrectly treats the background as part of the object.

### Score Jacobian Chaining [![arXiv](https://img.shields.io/badge/arXiv-2212.00774-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2212.00774)

**Results obtained by threestudio (Stable Diffusion)**

https://user-images.githubusercontent.com/19284678/236694871-87a247c1-2d3d-4cbf-89df-450bfeac3aca.mp4

Notable differences from the paper: N/A.

**Example running commands**

```sh
# train with sjc guidance in latent space
python launch.py --config configs/sjc.yaml --train --gpu 0 system.prompt_processor.prompt="A high quality photo of a delicious burger"
# train with sjc guidance in latent space, trump figure
python launch.py --config configs/sjc.yaml --train --gpu 0 system.prompt_processor.prompt="Trump figure" trainer.max_steps=30000 system.loss.lambda_emptiness="[15000,10000.0,200000.0,15001]" system.optimizer.params.background.lr=0.05 seed=42
```

**Tips**

- SJC uses subpixel rendering which decodes a `128x128` latent feature map for better visualization quality. You can turn off this feature by `system.subpixel_rendering=false` to save VRAM in validation/testing.

### Latent-NeRF [![arXiv](https://img.shields.io/badge/arXiv-2211.07600-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2211.07600)

**Results obtained by threestudio (Stable Diffusion)**

https://user-images.githubusercontent.com/19284678/236694876-5a270347-6a41-4429-8909-44c90c554e06.mp4

Notable differences from the paper: N/A.

We currently only implement Latent-NeRF for text-guided and Sketch-Shape for (text,shape)-guided 3D generation. Latent-Paint is not implemented yet.

**Example running commands**

```sh
# train Latent-NeRF in Stable Diffusion latent space
python launch.py --config configs/latentnerf.yaml --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger"
# refine Latent-NeRF in RGB space
python launch.py --config configs/latentnerf-refine.yaml --train --gpu 0 system.prompt_processor.prompt="a delicious hamburger" system.weights=path/to/latent/stage/trial/ckpts/last.ckpt

# train Sketch-Shape in Stable Diffusion latent space
python launch.py --config configs/sketchshape.yaml --train --gpu 0 system.guide_shape=load/shapes/teddy.obj system.prompt_processor.prompt="a teddy bear in a tuxedo"
# refine Sketch-Shape in RGB space
python launch.py --config configs/sketchshape-refine.yaml --train --gpu 0 system.guide_shape=load/shapes/teddy.obj system.prompt_processor.prompt="a teddy bear in a tuxedo" system.weights=path/to/latent/stage/trial/ckpts/last.ckpt
```

### Fantasia3D [![arXiv](https://img.shields.io/badge/arXiv-2303.13873-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2303.13873)

**Results obtained by threestudio (Stable Diffusion)**

https://user-images.githubusercontent.com/19284678/236694880-33b0db21-4530-47f1-9c3b-c70357bc84b3.mp4

Notable differences from the paper: N/A.

We currently only implement the geometry stage of Fantasia3D.

**Example running commands**

```sh
python launch.py --config configs/fantasia3d.yaml --train --gpu 0 system.prompt_processor.prompt="a DSLR photo of an ice cream sundae"
# Fantasia3D highly relies on the initialized SDF shape
# the default shape is a sphere with radius 0.5
# change the shape initialization to match your input prompt
python launch.py --config configs/fantasia3d.yaml --train --gpu 0 system.prompt_processor.prompt="The leaning tower of Pisa" system.geometry.shape_init=ellipsoid system.geometry.shape_init_params="[0.3,0.3,0.8]"
```

**Tips**

- If you find the shape easily diverge in early training stages, you may use a lower guidance scale by setting `system.guidance.guidance_scale=30.`.

### ProlificDreamer [![arXiv](https://img.shields.io/badge/arXiv-2305.16213-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2305.16213)

**Results obtained by threestudio (Stable Diffusion, 64x64, 15000 iterations)**

https://github.com/threestudio-project/threestudio/assets/19284678/29ded457-60a9-4d7e-8a5e-ab71a7ef361a

**IMPORTANT NOTE: This is an unofficial experimental implementation! The quality is still far from the paper. Please refer to [https://github.com/thu-ml/prolificdreamer](https://github.com/thu-ml/prolificdreamer) for official code release.**

We currently only experiment on the first stage (NeRF training), although the third stage is already implemented (mesh texture refinement), and the second stage is easy to implement too (mesh normal optimization). Some other important design factors that are not implemented:

- multiple particles

```sh
# object geneartion with 64x64 NeRF rendering, ~14GB VRAM
python launch.py --config configs/prolificdreamer.yaml --train --gpu 0 system.prompt_processor.prompt="a DSLR photo of a delicious croissant" data.width=64 data.height=64
# object generation with 512x512 NeRF rendering (original paper), >24GB VRAM
python launch.py --config configs/prolificdreamer.yaml --train --gpu 0 system.prompt_processor.prompt="a DSLR photo of a delicious croissant" data.width=512 data.height=512
# scene generation
python launch.py --config configs/prolificdreamer-scene.yaml --train --gpu 0 system.prompt_processor.prompt="Inside of a smart home, realistic detailed photo, 4k" data.width=64 data.height=64
```

### More to come, please stay tuned.

- [ ] [Dream3D](https://bluestyle97.github.io/dream3d/) [![arXiv](https://img.shields.io/badge/arXiv-2212.14704-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2212.14704)
- [ ] [DreamAvatar](https://yukangcao.github.io/DreamAvatar/) [![arXiv](https://img.shields.io/badge/arXiv-2304.00916-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2304.00916)

**If you would like to contribute a new method to threestudio, see [here](https://github.com/threestudio-project/threestudio#contributing-to-threestudio).**

## Prompt Library

For easier comparison, we collect the 397 preset prompts from the website of [DreamFusion](https://dreamfusion3d.github.io/gallery.html) in [this file](https://github.com/threestudio-project/threestudio/blob/main/load/prompt_library.json). You can use these prompts by setting `system.prompt_processor.prompt=lib:keyword1_keyword2_..._keywordN`. Note that the prompt should starts with `lib:` and all the keywords are separated by `_`. The prompt processor will match the keywords to all the prompts in the library, and will only succeed if there's **exactly one match**. The used prompt will be printed to console. Also note that you can't use this syntax to point to every prompt in the library, as there are prompts that are subset of other prompts lmao. We will enhance the use of this feature.

## Tips on Improving Quality

It's important to note that existing techniques that lift 2D T2I models to 3D cannot consistently produce satisfying results. Results from the great papers like DreamFusion and Magic3D are (to some extend) cherry-pickled, so don't be frustrated if you did not get what you expected on your first trial. Here are some tips that may help you improve the generation quality:

- **Increase batch size**. Large batch sizes help convergence and improve the 3D consistency of the geometry. State-of-the-art methods claims using large batch sizes: DreamFusion uses a batch size of 4; Magic3D uses a batch size of 32; Fantasia3D uses a batch size of 24; some results shown above uses a batch size of 8. You can easily change the batch size by setting `data.batch_size=N`. Increasing the batch size requires more VRAM. If you have limited VRAM but still want the benefit of large batch sizes, you may use [gradient accumulation provided by PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#accumulate-gradients) by setting `trainer.accumulate_grad_batches=N`. This will accumulate the gradient of several batches and achieve a large effective batch size. Note that if you use gradient accumulation, you may need to multiply all step values by N times in your config, such as values that have the name `X_steps` and `trainer.val_check_interval`, since now N batches equal to a large batch.
- **Train longer.** This helps if you can already obtain reasonable results and would like to enhance the details. If the result is still a mess after several thousand steps, training for a longer time often won't help. You can set the total training iterations by `trainer.max_steps=N`.
- **Try different seeds.** This is a simple solution if your results have correct overall geometry but suffer from the multi-face Janus problem. You can change the seed by setting `seed=N`. Good luck!
- **Tuning regularization weights.** Some methods have regularizaion terms which can be essential to obtaining good geometry. Try tuning the weights of these regularizations by setting `system.loss.lambda_X=value`. The specific values depend on your situation, you may refer to [tips for each supported model](https://github.com/threestudio-project/threestudio#supported-models) for more detailed instructions.

## VRAM Optimization

If you encounter CUDA OOM error, try the following in order (roughly sorted by recommendation) to meet your VRAM requirement.

- If you only encounter OOM at validation/test time, you can set `system.cleanup_after_validation_step=true` and `system.cleanup_after_test_step=true` to free memory after each validation/test step. This will slow down validation/testing.
- Use a smaller batch size or use gradient accumulation as demonstrated [here](https://github.com/threestudio-project/threestudio#tips-on-improving-quality).
- If you are using PyTorch1.x, enable [memory efficient attention](https://huggingface.co/docs/diffusers/optimization/fp16#memory-efficient-attention) by setting `system.guidance.enable_memory_efficient_attention=true`. PyTorch2.0 has built-in support for this optimization and is enabled by default.
- Enable [attention slicing](https://huggingface.co/docs/diffusers/optimization/fp16#sliced-attention-for-additional-memory-savings) by setting `system.guidance.enable_attention_slicing=true`. This will slow down training by ~20%.
- If you are using StableDiffusionGuidance, you can use [Token Merging](https://github.com/dbolya/tomesd) to **drastically** speed up computation and save memory. You can easily enable Token Merging by setting `system.guidance.token_merging=true`. You can also customize the Token Merging behavior by setting the parameters [here](https://github.com/dbolya/tomesd/blob/main/tomesd/patch.py#L183-L213) to `system.guidance.token_merging_params`. Note that Token Merging may degrade generation quality.
- Enable [sequential CPU offload](https://huggingface.co/docs/diffusers/optimization/fp16#offloading-to-cpu-with-accelerate-for-memory-savings) by setting `system.guidance.enable_sequential_cpu_offload=true`. This could save a lot of VRAM but will make the training **extremely slow**.

## Documentation

threestudio use [OmegaConf](https://github.com/omry/omegaconf) to manage configurations. You can literally change anything inside the yaml configuration file or by adding command line arguments without `--`. We list all arguments that you can change in the configuration in our [documentation](https://github.com/threestudio-project/threestudio/blob/main/DOCUMENTATION.md). Happy experimenting!

## Contributing to threestudio

- Fork the repository and create your branch from `main`.
- Install development dependencies:

```sh
pip install -r requirements-dev.txt
```

- If you are using VSCode as the text editor: (1) Install `editorconfig` extension. (2) Set the default linter to mypy to enable static type checking. (3) Set the default formatter to black. You could either manually format the document or let the editor format the document each time it is saved by setting `"editor.formatOnSave": true`.

- Run `pre-commit install` to install pre-commit hooks which will automatically format the files before commit.

- Make changes to the code, update README and DOCUMENTATION if needed, and open a pull request.

### Code Structure

Here we just briefly introduce the code structure of this project. We will make more detailed documentation about this in the future.

- All methods are implemented as a subclass of `BaseSystem` (in `systems/base.py`). There typically are six modules inside a system: geometry, material, background, renderer, guidance, and prompt_processor. All modules are subclass of `BaseModule` (in `utils/base.py`) except for guidance, and prompt_processor, which are subclass of `BaseObject` to prevent them from being treated as model parameters and better control their behavior in multi-GPU settings.
- All systems, modules, and data modules have their configurations in their own dataclasses.
- Base configurations for the whole project can be found in `utils/config.py`. In the `ExperimentConfig` dataclass, `data`, `system`, and module configurations under `system` are parsed to configurations of each class mentioned above. These configurations are strictly typed, which means you can only use defined properties in the dataclass and stick to the defined type of each property. This configuration paradigm (1) natually supports default values for properties; (2) effectively prevents wrong assignments of these properties (say typos in the yaml file) or inappropriate usage at runtime.
- This projects use both static and runtime type checking. For more details, see `utils/typing.py`.
- To update anything of a module at each training step, simply make it inherit to `Updateable` (see `utils/base.py`). At the beginning of each iteration, an `Updateable` will update itself, and update all its attributes that are also `Updateable`. Note that subclasses of `BaseSystem`, `BaseModule` and `BaseObject` are by default inherit to `Updateable`.

## Known Problems

- Gradients of Vanilla MLP parameters are empty in AMP (temporarily fixed by disabling autocast).
- FullyFused MLP may cause NaNs in 32 precision.

## Credits

threestudio is built on the following amazing open-source projects:

- **[Lightning](https://github.com/Lightning-AI/lightning)** Framework for creating highly organized PyTorch code.
- **[OmegaConf](https://github.com/omry/omegaconf)** Flexible Python configuration system.
- **[NerfAcc](https://github.com/KAIR-BAIR/nerfacc)** Plug-and-play NeRF acceleration.

The following repositories greatly inspire threestudio:

- **[Stable-DreamFusion](https://github.com/ashawkey/stable-dreamfusion)**
- **[Latent-NeRF](https://github.com/eladrich/latent-nerf)**
- **[Score Jacobian Chaining](https://github.com/pals-ttic/sjc)**
- **[Fantasia3D.unofficial](https://github.com/ashawkey/fantasia3d.unofficial)**

Thanks to the maintainers of these projects for their contribution to the community!

## Citing threestudio

If you find threestudio helpful, please consider citing:

```
@Misc{threestudio2023,
  author =       {Yuan-Chen Guo and Ying-Tian Liu and Chen Wang and Zi-Xin Zou and Guan Luo and Chia-Hao Chen and Yan-Pei Cao and Song-Hai Zhang},
  title =        {threestudio: A unified framework for 3D content generation},
  howpublished = {\url{https://github.com/threestudio-project/threestudio}},
  year =         {2023}
}
```

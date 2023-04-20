## Setup
- Python >= 3.8
- PyTorch >= 1.12 (PyTorch 2.0 not tested)
- `pip install -r requirements.txt` (change torch source url and version accroding to your CUDA version, requires torch>=1.12)
- `pip install -r requirements-dev.txt` for linters and formatters, and set the default linter in vscode to mypy

## Known Problems
- Validation/testing using resumed checkpoints have iteration=0, will be problematic if some settings are step-dependent.
- Gradients of Vanilla MLP parameters are empty if autocast is enabled in AMP (temporarily fixed by disabling autocast).
- FullyFused MLP causes NaNs in 32 precision. Aggressive gradient clipping could solve the issue (e.g., `system.guidance.grad_clip=0.1`).

## Precision
- mixed precision training: `trainer.precision=16-mixed`; `system.guidance.half_precision_weights=true`; either `VanillaMLP` and `FullyFusedMLP` can be used.
- float32 precision training: `trainer.precision=32`; `system.guidance.half_precision_weights=false`; only `VanillaMLP` can be used.

## Structure
- All methods should be implemented as a subclass of `BaseSystem` (in `systems/base.py`). For the DreamFusion system, there're 6 modules: geometry, material, background, renderer, guidance, prompt_processor. All modules are subclass of `BaseModule` (in `utils/base.py`).
- All systems, modules, and data modules have their configurations in their own dataclass named `Config`.
- Base configurations for the whole project can be found in `utils/config.py`. In the `ExperimentConfig` dataclass, `data`, `system`, and module configurations under `system` are parsed to configurations of each class mentioned above. These configurations are strictly typed, which means you can only use defined properties in the dataclass and stick to the defined type of each property. This configuration paradigm is better than the one used in `instant-nsr-pl` as (1) it natually supports default values for properties; (2) it effectively prevents wrong assignments of these properties (say typos in the yaml file) and inappropriate usage at runtime.
- This projects use both static and runtime type checking. For more details, see `utils/typing.py`.

## Run
### DreamFusion
```bash
# train with diffuse material and point lighting
python launch.py --config configs/dreamfusion.yaml --train --gpu 0 system.prompt_processor.prompt="a hamburger"
# train with simple surface color without material assumption
python launch.py --config configs/dreamfusion-wonormal.yaml --train --gpu 0 system.prompt_processor.prompt="a hamburger"
```
### Latent-NeRF
```bash
# train in stable-diffusion latent space
python launch.py --config configs/latentnerf.yaml --train --gpu 0 system.prompt_processor.prompt="a hamburger"
# refine in RGB space
python launch.py --config configs/latentnerf-refine.yaml --train --gpu 0 system.prompt_processor.prompt="a hamburger" system.weights=path/to/latentnerf/weights
```
### Fantasia3D (WIP)
I by far have implemented the early training stage of Fantasia3D, which regards the downsampled normal and silhouette as the latent feature map and optimizes using SDS.
```bash
python launch.py --config configs/fantasia3d.yaml --train --gpu 0 system.prompt_processor.prompt="a ripe strawberry"
# Fantasia3D highly relies on the initialized SDF shape
# change the shape initialization to match your input prompt
python launch.py --config configs/fantasia3d.yaml --train --gpu 0 system.prompt_processor.prompt="The leaning tower of Pisa" system.geometry.shape_init=ellipsoid system.geometry.shape_init_params="[0.3,0.3,0.8]"
```

### Score Jacobian Chaining
```bash
# train with sjc guidance in latent space
python launch.py --config configs/sjc.yaml --train --gpu 0 system.prompt_processor.prompt="A high quality photo of a delicious burger" 
# train with sjc guidance in latent space, trump figure
python launch.py --config configs/sjc.yaml --train --gpu 0 system.prompt_processor.prompt="Trump figure" seed=10 system.renderer.num_samples_per_ray=512 trainer.max_steps=30000 system.loss.lambda_emptiness=[15000,10000.0,200000.0,15001]
```

### Image-Condition DreamFusion
```bash
# train with single image reference and stable-diffusion sds guidance
python launch.py --config configs/imagecondition.yaml --train --gpu 0
```

## Tips
- To resume a model and continue training, please load the `parsed.yaml` in the trial directory and set `resume` to the checkpoint path. Example:
```bash
python launch.py --config path/to/your/trial/output/parsed.yaml --train --gpu 0 resume=path/to/your/checkpoint
```
- Press ctrl+c **once** will stop training and continue to testing. Press ctrl+c the second time to fully quit the program.
- To update anything of a module at each training step, simply make it inherit to `Updateable` (see `utils/base.py`). At the beginning of each iteration, an `Updateable` will update itself, and update all its attributes that are also `Updateable`. Note that subclasses of `BaseSystem` and `BaseModule` (including all geometry, materials, guidance, prompt processors, and renderers) are by default inherit to `Updateable`.
- For easier comparison, we collect the 397 preset prompts from the website of [DreamFusion](https://dreamfusion3d.github.io/gallery.html). You can use these prompts by setting `system.prompt_processor.prompt=lib:keyword1_keyword2_..._keywordN`. Note that the prompt should starts with `lib:` and all the keywords are separated by `_`. The prompt processor will match the keywords to all the prompts in the library, and will only succeed if there's exactly one match. The used prompt will be printed to console.
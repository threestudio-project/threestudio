## Setup
- Python >= 3.8
- `pip install -r requirements.txt` (change torch source url and version accroding to your CUDA version, requires torch>=1.12)
- `pip install -r requirements-dev.txt` for linters and formatters, and set the default linter in vscode to mypy

## Known Problems
- Validation/testing using resumed checkpoints have iteration=0, will be problematic if some settings are step-dependent.
- Gradients of Vanilla MLP parameters are empty if autocast is enabled in AMP (temporarily fixed by disabling autocast).
- FullyFused MLP causes NaNs both in mixed-precision and 32 precision. Gradient clipping does not solve the issue.

## Structure
- All methods should be implemented as a subclass of `BaseSystem` (in `systems/base.py`). For the DreamFusion system, there're 6 modules: geometry, material, background, renderer, guidance, prompt_processor. All modules are subclass of `BaseModule` (in `utils/base.py`).
- All systems, modules, and data modules have their configurations in their own dataclass named `Config`.
- Base configurations for the whole project can be found in `utils/config.py`. In the `ExperimentConfig` dataclass, `data`, `system`, and module configurations under `system` are parsed to configurations of each class mentioned above. These configurations are strictly typed, which means you can only use defined properties in the dataclass and stick to the defined type of each property. This configuration paradigm is better than the one used in `instant-nsr-pl` as (1) it natually supports default values for properties; (2) it effectively prevents wrong assignments of these properties (say typos in the yaml file) and inappropriate usage at runtime.
- This projects use both static and runtime type checking. For more details, see `utils/typing.py`.

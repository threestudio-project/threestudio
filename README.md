## Setup
- Python >= 3.8
- `pip install -r requirements.txt` (change torch source url and version accroding to your CUDA version, requires torch>=1.12)
- `pip install -r requirements-dev.txt` for linters and formatters, and set the default linter in vscode to mypy

## Known Problems
- Saved images may be broken at the last training step and test steps after training. Testing using resumed checkpoints work fine.
- Validation/testing using resumed checkpoints have iteration=0, will be problematic if some settings are step-dependent.



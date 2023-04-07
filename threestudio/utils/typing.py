"""
This module contains type annotations for the project, using
1. Python type hints (https://docs.python.org/3/library/typing.html) for Python objects
2. jaxtyping (https://github.com/google/jaxtyping/blob/main/API.md) for PyTorch tensors

Two types of typing checking can be used:
1. Static type checking with mypy (install with pip and enabled as the default linter in VSCode)
2. Runtime type checking with typeguard (install with pip and triggered at runtime, mainly for tensor dtype and shape checking)
"""

# Basic types
from typing import Any, Optional, Union, Iterable, Callable, Type, Dict, List, Tuple, Literal, TypeVar, NewType

# Config type
from omegaconf import DictConfig

# PyTorch Tensor type
from torch import Tensor

# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import Shaped, Bool, Num, Inexact, Float, Complex, Integer, UInt, Int

# Runtime type checking decorator
from typeguard import typechecked as typechecker

"""Execution backends. Heavy ML dependencies are imported only when explicitly requested."""

from .local_model import LocalCausalLM
from .numpy_backend import NumpyLoRABackend

__all__ = ["NumpyLoRABackend", "LocalCausalLM"]

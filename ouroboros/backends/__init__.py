"""Execution backends. Heavy ML dependencies are imported only when explicitly requested."""

from .numpy_backend import NumpyLoRABackend

__all__ = ["NumpyLoRABackend"]

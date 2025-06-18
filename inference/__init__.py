"""Inference utilities for the trained DeepONet models."""

from .base_inference import BaseInference
from .methods_inference import MethodsInference


class Inference(BaseInference, MethodsInference):
    """Combines ``BaseInference`` with convenience methods."""

    pass

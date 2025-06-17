"""Rowdy activation used in the neural networks."""

from .base_activations import BaseRowdyActivation
from .methods_activations import MethodsRowdyActivation


class RowdyActivation(BaseRowdyActivation, MethodsRowdyActivation):
    """Scale a base activation function by a learnable factor ``alpha``."""

    def forward(self, x):
        return self.base_fn(x) * self.alpha


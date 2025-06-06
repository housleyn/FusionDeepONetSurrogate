from .base_activations import BaseRowdyActivation
from .methods_activations import MethodsRowdyActivation

class RowdyActivation(BaseRowdyActivation, MethodsRowdyActivation):
    def forward(self,x):
        return self.alpha * self.base_fn(x)
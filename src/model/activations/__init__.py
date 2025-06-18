from .base_activations import BaseRowdyActivation
from .methods_activations import MethodsRowdyActivation

class RowdyActivation(BaseRowdyActivation, MethodsRowdyActivation):
    def forward(self,x):
        return  self.base_fn(x) * self.alpha
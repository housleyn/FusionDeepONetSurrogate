from .base_activations import BaseRowdyActivation


class RowdyActivation(BaseRowdyActivation):
    def forward(self,x):
        return  self.base_fn(x) * self.alpha
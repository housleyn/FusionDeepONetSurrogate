from .base_MLP import BaseMLP 
from .methods_MLP import MethodsMLP

class MLP(BaseMLP, MethodsMLP):
    def forward(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return self.final(x)
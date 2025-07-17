from .base_MLP import BaseMLP 


class MLP(BaseMLP):
    def forward(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return self.final(x)
    
    def forward_with_outputs(self, x):
        hidden_outputs = []
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
            hidden_outputs.append(x)
        return self.final(x), hidden_outputs
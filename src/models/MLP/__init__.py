from .base_MLP import BaseMLP 

class MLP(BaseMLP):
    def forward(self, x):
        for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
            x = act(layer(x))
            if i > 0:
                x = self.dropouts[i-1](x)
        return self.final(x)
    
    def forward_with_outputs(self, x):
        hidden_outputs = []
        for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
            x = act(layer(x))
            if i > 0:
                x = self.dropouts[i-1](x)
            hidden_outputs.append(x)
        return self.final(x), hidden_outputs
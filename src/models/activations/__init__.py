from .base_activations import BaseRowdyActivation
import torch
import torch.nn as nn

class RowdyActivation(BaseRowdyActivation):
    def forward(self,x):
        return  self.base_fn(x) * self.alpha



class HarmonicActivation(nn.Module):

    def __init__(self, size):
        super().__init__()
        
        # tanh amplitude
        self.alpha = nn.Parameter(torch.ones(size))
        
        # sine amplitude
        self.beta = nn.Parameter(torch.zeros(size))
        
        # sine frequency
        self.freq = nn.Parameter(torch.ones(size) * 0.1)

        # optional bias (matches their c and c1)
        self.bias_tanh = nn.Parameter(torch.zeros(size))
        self.bias_sin = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        return (
            self.alpha * torch.tanh(x + self.bias_tanh)
            + self.beta * torch.sin(self.freq * x + self.bias_sin)
        )

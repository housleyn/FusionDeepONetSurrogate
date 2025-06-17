import torch
import torch.nn as nn

class BaseRowdyActivation(nn.Module):
    def __init__(self, size, base_fn=torch.tanh):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(size))
        self.base_fn = base_fn

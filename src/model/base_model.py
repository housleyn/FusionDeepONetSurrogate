import torch
import torch.nn as nn
from .MLP import MLP
from .activations import RowdyActivation

class BaseModel(nn.Module):
    def __init__(self, coord_dim, param_dim, hidden_size, num_hidden_layers, out_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers



        self.branch = MLP(param_dim, hidden_size * out_dim, hidden_size, num_hidden_layers)

        self.trunk_layers = nn.ModuleList([                                               #eq 4.2
            nn.Linear(coord_dim if i == 0 else hidden_size, hidden_size)
            for i in range(num_hidden_layers)
        ])
        self.trunk_activations = nn.ModuleList([
            RowdyActivation(hidden_size) for _ in range(num_hidden_layers)
        ])

        self.trunk_final = nn.Linear(hidden_size, hidden_size)
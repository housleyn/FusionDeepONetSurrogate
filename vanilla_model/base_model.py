import torch.nn as nn
from model.MLP import MLP
from model.activations import RowdyActivation

class BaseModel(nn.Module):
    def __init__(self, coord_dim, param_dim, hidden_size, num_hidden_layers, out_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers

        self.branch = MLP(param_dim, hidden_size * out_dim, hidden_size, num_hidden_layers)

        self.trunk = MLP(coord_dim, hidden_size, hidden_size, num_hidden_layers)
        self.trunk_final = nn.Linear(hidden_size, hidden_size)

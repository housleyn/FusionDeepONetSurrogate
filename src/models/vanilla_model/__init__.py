import torch
import torch.nn as nn
from ..MLP import MLP


class VanillaDeepONet(nn.Module):
    def __init__(self, coord_dim, param_dim, hidden_size, num_hidden_layers, out_dim, aux=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.aux = aux
        self.branch = MLP(param_dim, hidden_size * out_dim, hidden_size, num_hidden_layers)
        self.trunk = MLP(coord_dim, hidden_size, hidden_size, num_hidden_layers)

    def forward(self, coords, params, sdf, aux=None):
        Batch_size, n_pts, _ = coords.shape
        branch_out = self.branch(params)
        trunk_out = self.trunk(coords)
        branch_out = branch_out.view(Batch_size, self.out_dim, self.hidden_size)
        trunk_out = trunk_out.view(Batch_size, n_pts, 1, self.hidden_size)
        branch_out = branch_out.unsqueeze(1)
        out = torch.sum(branch_out * trunk_out, dim=-1)
        return out

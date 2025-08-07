import torch
import torch.nn as nn
from ..MLP import MLP


class VanillaDeepONet(nn.Module):
    def __init__(self, coord_dim, param_dim, hidden_size, num_hidden_layers, out_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers

        self.branch = MLP(param_dim, hidden_size * out_dim, hidden_size, num_hidden_layers)

        self.trunk = MLP(coord_dim, hidden_size, hidden_size, num_hidden_layers)

    def forward(self, coords, params, sdf):
        B, n_pts, _ = coords.shape
        H = self.hidden_size
        O = self.out_dim

        branch_out = self.branch(params)
        trunk_out = self.trunk(coords)

        branch_out = branch_out.view(B, O, H)
        trunk_out = trunk_out.view(B, n_pts, 1, H)
        branch_out = branch_out.unsqueeze(1)

        out = torch.sum(branch_out * trunk_out, dim=-1)
        return out

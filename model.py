import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, num_hidden_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = []

        self.layers.append(nn.Linear(in_features, hidden_features))
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
        self.final = nn.Linear(hidden_features, out_features)


    def forward_with_activations(self,x):
        activations = []
        for layer in self.layers:
            x = torch.tanh(layer(x))
            activations.append(x)
        return self.final(x), activations
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.final(x)
    
class FusionDeepONet(nn.Module):
    def __init__(self, coord_dim, param_dim, hidden_size, num_hidden_layers, out_dim):
        super().__init__()
        self.branch = MLP(param_dim, hidden_size * out_dim, hidden_size, num_hidden_layers)
        self.trunk_layers = nn.ModuleList([
            nn.Linear(coord_dim + hidden_size, hidden_size)  # fused input: coord + branch
            if i == 0 else
            nn.Linear(hidden_size + hidden_size, hidden_size)  # hidden + branch
            for i in range(num_hidden_layers)
        ])
        self.final = nn.Linear(hidden_size, out_dim)

    def forward(self, coords, params):
        # coords: (batch, n_pts, coord_dim)
        # params: (batch, param_dim)

        batch_size, n_pts, _ = coords.shape
        out_dim = self.final.out_features

        # Expand params to match n_pts
        params_expanded = params.unsqueeze(1).repeat(1, n_pts, 1)  # (B, n_pts, param_dim)

        # Compute branch output and activations
        branch_out, branch_acts = self.branch.forward_with_activations(params.view(-1, params.shape[-1]))
        branch_acts = [b.unsqueeze(1).repeat(1, n_pts, 1) for b in branch_acts]  # match coords
        branch_out = branch_out.view(batch_size, out_dim, -1)  # shape: (B, out_dim, H)

        # Trunk forward pass with fusion
        x = coords
        for i, layer in enumerate(self.trunk_layers):
            branch_fuse = branch_acts[i]
            x = torch.cat([x, branch_fuse], dim=-1)
            x = torch.tanh(layer(x))

        out = self.final(x)  # (B, n_pts, out_dim)
        return out

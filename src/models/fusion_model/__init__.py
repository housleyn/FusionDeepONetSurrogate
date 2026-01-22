import torch
import torch.nn as nn
from ..MLP import MLP
from ..activations import RowdyActivation
from ..activations import HarmonicActivation
class FusionDeepONet(nn.Module):
    def __init__(self, coord_dim, param_dim, hidden_size, num_hidden_layers, out_dim, aux_dim=0, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.branch = MLP(param_dim, hidden_size * out_dim, hidden_size, num_hidden_layers, dropout)
        input_dim = coord_dim + aux_dim   
        self.trunk_layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_size, hidden_size)
            for i in range(num_hidden_layers)
        ])
        self.trunk_activations = nn.ModuleList([
            HarmonicActivation(hidden_size) for _ in range(num_hidden_layers)
        ])
        self.trunk_dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_hidden_layers - 1)
        ])
        self.trunk_final = nn.Linear(hidden_size, hidden_size)

    def forward(self, coords, params, sdf, aux=None):
        Batch_size, n_pts, _ = coords.shape 
        branch_out, branch_hiddens = self.branch.forward_with_outputs(params)
        skip = []
        for i in range(len(branch_hiddens)):
            if i == 0:
                skip.append(branch_hiddens[i])
            else:
                skip.append(branch_hiddens[i] + skip[i-1])
        
        x = torch.cat((coords, sdf), dim=-1) if aux is None else torch.cat((coords, sdf, aux), dim=-1)
        for i, (layer,act) in enumerate(zip(self.trunk_layers,self.trunk_activations)):
            x = act(layer(x))
            if i > 0:
                x = self.trunk_dropouts[i-1](x)
            if i < len(skip):
                x = x * skip[i].unsqueeze(1)                                                         
        trunk_features = self.trunk_final(x) 
        branch_coefficients = branch_out.view(Batch_size, self.out_dim, self.hidden_size)  
        trunk_features = trunk_features.view(Batch_size, n_pts, 1, self.hidden_size)  
        branch_coefficients = branch_coefficients.unsqueeze(1)  
        out = torch.sum (branch_coefficients * trunk_features, dim=-1)  
        return out
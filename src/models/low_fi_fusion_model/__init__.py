import torch
import torch.nn as nn
from ..MLP import MLP
from ..activations import RowdyActivation
import numpy as np

class Low_Fidelity_FusionDeepONet(nn.Module):
    def __init__(self, coord_dim, param_dim, hidden_size, num_hidden_layers, out_dim, npz_path, aux_dim=0, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_dim = out_dim 
        self.num_hidden_layers = num_hidden_layers
        self.outputs_mean = self._load_stats(npz_path)["outputs_mean"]
        self.outputs_std = self._load_stats(npz_path)["outputs_std"]
        self.branch = MLP(param_dim, hidden_size * out_dim, hidden_size, num_hidden_layers, dropout)
        input_dim = coord_dim + aux_dim  
        self.trunk_layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_size, hidden_size)
            for i in range(num_hidden_layers)
        ])
        self.trunk_activations = nn.ModuleList([
            RowdyActivation(hidden_size) for _ in range(num_hidden_layers)
        ])
        self.trunk_dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_hidden_layers - 1)
        ])
        self.trunk_final = nn.Linear(hidden_size, hidden_size)

    def forward(self, coords, params, sdf, aux=None):
        Batch_size, n_pts, _ = coords.shape 
        branch_out, branch_hiddens = self.branch.forward_with_outputs(params)
        fusion_gate = [branch_hiddens[0]]                                                      
        for i in range(1, self.num_hidden_layers): 
            fusion_gate.append(branch_hiddens[i] + fusion_gate[-1])
        x = torch.cat((coords, sdf), dim=-1) if aux is None else torch.cat((coords, sdf, aux), dim=-1)
        for i, (layer,act) in enumerate(zip(self.trunk_layers,self.trunk_activations)):
            x = act(layer(x))
            if i > 0:
                x = self.trunk_dropouts[i-1](x)
            if i < len(fusion_gate):
                fusion_gate_i = fusion_gate[i].unsqueeze(1) 
                x = x * fusion_gate_i                                                         
        trunk_features = self.trunk_final(x) 
        branch_coefficients = branch_out.view(Batch_size, self.out_dim, self.hidden_size)  
        trunk_features = trunk_features.view(Batch_size, n_pts, 1, self.hidden_size)  
        branch_coefficients = branch_coefficients.unsqueeze(1)  
        out = torch.sum (branch_coefficients * trunk_features, dim=-1)  
        return out
    
    def _load_stats(self, npz_path):
        data = np.load(npz_path)
        return {
            "outputs_mean": torch.tensor(data["outputs_mean"], dtype=torch.float32),
            "outputs_std": torch.tensor(data["outputs_std"], dtype=torch.float32),
        }
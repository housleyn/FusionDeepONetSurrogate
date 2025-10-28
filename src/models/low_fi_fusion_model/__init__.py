import torch
import torch.nn as nn
from ..MLP import MLP
from ..activations import RowdyActivation
import numpy as np

class Low_Fidelity_FusionDeepONet(nn.Module):
    def __init__(self, coord_dim, param_dim, hidden_size, num_hidden_layers, out_dim, npz_path, aux_dim=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_dim = out_dim 
        
        self.num_hidden_layers = num_hidden_layers
        self.outputs_mean = self._load_stats(npz_path)["outputs_mean"]
        self.outputs_std = self._load_stats(npz_path)["outputs_std"]
        

        self.branch = MLP(param_dim, hidden_size * out_dim, hidden_size, num_hidden_layers)

        input_dim = coord_dim + aux_dim #coord_dim + distance_dim + free stream values (6)  
        self.trunk_layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_size, hidden_size)
            for i in range(num_hidden_layers)
        ])
        self.trunk_activations = nn.ModuleList([
            RowdyActivation(hidden_size) for _ in range(num_hidden_layers)
        ])
        self.trunk_final = nn.Linear(hidden_size, hidden_size)

    def forward(self, coords, params, sdf, aux=None):

        B, n_pts, _ = coords.shape #batch
        H = self.hidden_size #hidden
        O = self.out_dim #output

        branch_out, branch_hiddens = self.branch.forward_with_outputs(params)
        
        S = [branch_hiddens[0]]                                                      #eq. 4.3
        for i in range(1, self.num_hidden_layers): #trunk fusion forloop
            S.append(branch_hiddens[i] + S[-1])

        x = torch.cat((coords, sdf), dim=-1) if aux is None else torch.cat((coords, sdf, aux), dim=-1)
        for i, (layer,act) in enumerate(zip(self.trunk_layers,self.trunk_activations)):
            x = act(layer(x))

            if i < len(S):
                S_i = S[i].unsqueeze(1) #change dimension of branch to be multiplied by trunk (B, 1, H)
                x = x* S_i #element wise multiplication                                                        eq. 4.4
            
            
        YL = self.trunk_final(x) #(B, N_pts, H)                                                        eq 4.5
        ZL = branch_out.view(B, O, H)  # (B, O, H)
        YL = YL.view(B, n_pts, 1, H)  # (B, N_pts, 1, H)
        ZL = ZL.unsqueeze(1)  # (B, 1, O, H)

        out = torch.sum (ZL * YL, dim=-1)  # (B, N_pts, O)                                              dot product
        return out
    
   
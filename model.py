import torch
import torch.nn as nn
import torch.nn.functional as F


class RowdyActivation(nn.Module):  #                                                           eq. 4.1
    def __init__(self, size, base_fn=torch.tanh):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(size))
        self.base_fn = base_fn 

    def forward(self,x):
        return self.alpha * self.base_fn(x)

class MLP(nn.Module):            #                                                             eq 4.1
    def __init__(self, in_features, out_features, hidden_features, num_hidden_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        self.layers.append(nn.Linear(in_features, hidden_features))
        self.activations.append(RowdyActivation(hidden_features))

        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
            self.activations.append(RowdyActivation(hidden_features))
        self.final = nn.Linear(hidden_features, out_features)


    def forward_with_outputs(self,x):
        hidden_outputs = []
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
            hidden_outputs.append(x)
        return self.final(x), hidden_outputs
    
    def forward(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return self.final(x)
    
class FusionDeepONet(nn.Module):
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

    def forward(self, coords, params):
        # coords: (batch, n_pts, coord_dim)
        # params: (batch, param_dim)

        B, n_pts, _ = coords.shape #batch
        H = self.hidden_size #hidden
        O = self.out_dim #output

        branch_out, branch_hiddens = self.branch.forward_with_outputs(params)
        
        S = [branch_hiddens[0]]                                                      #eq. 4.3
        for i in range(1, self.num_hidden_layers): #trunk fusion forloop
            S.append(branch_hiddens[i] + S[-1])

        x = coords 
        for i, (layer,act) in enumerate(zip(self.trunk_layers,self.trunk_activations)):
            x = layer(x)

            if i < len(S):
                S_i = S[i].unsqueeze(1) #change dimension of branch to be multiplied by trunk (B, 1, H)
                x = x* S_i #element wise multiplication                                                        eq. 4.4
            
            x = act(x)
        

        YL = self.trunk_final(x) #(B, N_pts, H)                                                        eq 4.5
        ZL = branch_out.view(B, O, H)  # (B, O, H)
        YL = YL.view(B, n_pts, 1, H)  # (B, N_pts, 1, H)
        ZL = ZL.unsqueeze(1)  # (B, 1, O, H)

        out = torch.sum (ZL * YL, dim=-1)  # (B, N_pts, O)                                              dot product
        return out

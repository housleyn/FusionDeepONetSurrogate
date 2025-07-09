from .base_model import BaseModel
from .methods_model import MethodsModel 
import torch

class FusionDeepONet(BaseModel, MethodsModel):

    def forward(self, coords, params, sdf):
        # coords: (batch, n_pts, coord_dim)
        # params: (batch, param_dim)

        B, n_pts, _ = coords.shape #batch
        H = self.hidden_size #hidden
        O = self.out_dim #output

        branch_out, branch_hiddens = self.branch.forward_with_outputs(params)
        
        S = [branch_hiddens[0]]                                                      #eq. 4.3
        for i in range(1, self.num_hidden_layers): #trunk fusion forloop
            S.append(branch_hiddens[i] + S[-1])

        x = torch.cat((coords, sdf), dim=-1) 
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
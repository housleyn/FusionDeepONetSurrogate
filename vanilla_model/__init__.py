import torch
from .base_model import BaseModel
from .methods_model import MethodsModel

class VanillaDeepONet(BaseModel, MethodsModel):
    def forward(self, coords, params):
        B, n_pts, _ = coords.shape
        H = self.hidden_size
        O = self.out_dim

        branch_out = self.branch(params)

        x = coords
        for layer, act in zip(self.trunk_layers, self.trunk_activations):
            x = act(layer(x))
        trunk_out = self.trunk_final(x)

        branch_out = branch_out.view(B, O, H)
        trunk_out = trunk_out.view(B, n_pts, 1, H)
        branch_out = branch_out.unsqueeze(1)

        out = torch.sum(branch_out * trunk_out, dim=-1)
        return out

import torch
from .base_model import BaseModel


class VanillaDeepONet(BaseModel):
    def forward(self, coords, params):
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

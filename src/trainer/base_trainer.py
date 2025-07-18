import torch
import torch.nn as nn
from .gradient_weighting import gradient_weighted_value_loss

class BaseTrainer:
    def __init__(self, model, dataloader, device="cpu", lr=1e-3, project_name="project", pressure_idx=-1):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.criterion = gradient_weighted_value_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.pressure_idx = pressure_idx

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.75)
        self.project_name = project_name



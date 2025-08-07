import torch
import torch.nn as nn


class BaseTrainer:
    def __init__(self, model, dataloader, device="cpu", lr=1e-3, lr_gamma=1.5, project_name="project", loss_type="mse"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_type = loss_type
        

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_gamma)
        self.project_name = project_name



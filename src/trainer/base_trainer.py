import torch
import torch.nn as nn

class BaseTrainer:
    def __init__(self, model, dataloader, device="cpu", lr=1e-3, project_name="project"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.75)
        self.project_name = project_name



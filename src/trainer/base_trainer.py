import torch
import torch.nn as nn


class BaseTrainer:
    def __init__(self, model, dataloader, device="cpu", lr=1e-3, lr_gamma=1.5, project_name="project"):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.decay_rate = .91
        self.decay_steps = 1000
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_gamma)
        self.project_name = project_name
        self.global_step = 0



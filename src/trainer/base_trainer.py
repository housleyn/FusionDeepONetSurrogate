import torch
import torch.nn as nn


class BaseTrainer:
    def __init__(self, model, dataloader, device="cpu", lr=1e-3, lr_gamma=1.5, project_name="project", use_amp=True, amp_dtype=torch.float16):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_gamma)
        self.project_name = project_name

        # NEW
        self.use_amp = use_amp and ("cuda" in str(device))
        self.amp_dtype = amp_dtype
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)


import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    def __init__(self, model, dataloader, device="cpu", lr=1e-3):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

    def train(self, num_epochs=1000, print_every=100):
        self.model.train()
        loss_history = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for coords, params, targets in self.dataloader:
                coords = coords.to(self.device)       # shape: (B, N, 3)
                params = params.to(self.device)       # shape: (B, 1)
                targets = targets.to(self.device)     # shape: (B, N, 5)

                self.optimizer.zero_grad()
                outputs = self.model(coords, params)  # shape: (B, N, output_dim)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            self.lr_scheduler.step()
            avg_loss = epoch_loss / len(self.dataloader)
            loss_history.append(avg_loss)

            if epoch % print_every == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f} | LR: {self.lr_scheduler.get_last_lr()[0]:.2e}")

        return loss_history

    def save_model(self, path="fusion_deeponet.pt"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

import torch
import os

def to_device_batch(self, batch):
    coords, params, targets, sdf, *maybe_aux = batch
    coords  = coords.to(self.device)
    params  = params.to(self.device)
    targets = targets.to(self.device)
    sdf     = sdf.to(self.device)
    aux = maybe_aux[0].to(self.device) if len(maybe_aux) else None
    return coords, params, targets, sdf, aux

def train_one_epoch(self, train_loader):
    self.model.train()
    epoch_loss, total_samples = 0.0, 0

    for batch in train_loader:
        coords, params, targets, sdf, aux = to_device_batch(self, batch)

        self.optimizer.zero_grad()
        outputs = self.model(coords, params, sdf, aux=aux)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

        bs = targets.size(0)
        epoch_loss += loss.item() * bs
        total_samples += bs

    return epoch_loss / total_samples

def save_best_checkpoint_if_needed(self, epoch, test_loss, best_loss):
    if test_loss >= best_loss:
        return best_loss

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "scheduler_state_dict": self.lr_scheduler.state_dict(),
        "loss": test_loss,
    }
    torch.save(checkpoint, f"Outputs/{self.project_name}/checkpoints/best_model.pt")
    return test_loss

def print_epoch(self, epoch, avg_loss, test_loss, best_loss, epoch_time, avg_time):
    print(
        f"Epoch {epoch:4d} | Train Loss: {avg_loss:.6f} | Test Loss: {test_loss:.6f}"
        f"| Best Test Loss: {best_loss:.6f} | LR: {self.lr_scheduler.get_last_lr()[0]:.2e} "
        f"| Epoch Time: {epoch_time:.2f}s"
        f"| Avg Time: {avg_time:.2f}s"
    )

def load_best_weights(self, path):
    checkpoint = torch.load(path, map_location=self.device)
    self.model.load_state_dict(checkpoint["model_state_dict"])
    self.model.eval()
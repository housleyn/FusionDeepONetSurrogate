import torch
import os
import time
from .train_helpers import (to_device_batch, train_one_epoch, save_best_checkpoint_if_needed, print_epoch, load_best_weights)

class MethodsTrainer:
    
    def train(self, train_loader, test_loader, num_epochs, print_every):
        loss_history, test_loss_history, epoch_times = [], [], []
        best_loss = float("inf")
        os.makedirs(f"Outputs/{self.project_name}/checkpoints", exist_ok=True)

        for epoch in range(num_epochs):
            t0 = time.time()

            avg_loss = train_one_epoch(self, train_loader)

            if (epoch + 1) % 10000 == 0:
                self.lr_scheduler.step()

            test_loss = self.evaluate(test_loader)
            best_loss = save_best_checkpoint_if_needed(self, epoch, test_loss, best_loss)

            loss_history.append(avg_loss)
            test_loss_history.append(test_loss)

            epoch_time = time.time() - t0
            epoch_times.append(epoch_time)

            if epoch % print_every == 0 or epoch == num_epochs - 1:
                print_epoch(self,
                    epoch, avg_loss, test_loss, best_loss,
                    epoch_time, sum(epoch_times) / len(epoch_times)
                )

        return loss_history, test_loss_history

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                coords, params, targets, sdf, aux = to_device_batch(self, batch)
                outputs = self.model(coords, params, sdf, aux=aux)
                loss = self.criterion(outputs, targets)
                bs = targets.size(0)
                total_loss += loss.item() * bs
                total_samples += bs
        return total_loss / total_samples


    def save_model(self, path=None, low_fi=False):
        load_best_weights(self,f'Outputs/{self.project_name}/checkpoints/best_model.pt')
        if path is None:
            os.makedirs(f'Outputs/{self.project_name}/model', exist_ok=True)
            if low_fi:
                path = f'Outputs/{self.project_name}/model/low_fi_fusion_deeponet.pt'
            else:
                path = f'Outputs/{self.project_name}/model/fusion_deeponet.pt'
        torch.save(self.model.state_dict(), path)


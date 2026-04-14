import torch
import os
import time
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from .train_helpers import (to_device_batch, train_one_epoch, save_best_checkpoint_if_needed, print_epoch, load_best_weights)

class MethodsTrainer:
    
    def train(self, train_loader, test_loader, num_epochs, print_every):
        loss_history, test_loss_history, epoch_times = [], [], []
        best_loss = float("inf")
        os.makedirs(f"Outputs/{self.project_name}/checkpoints", exist_ok=True)

        for epoch in range(num_epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
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
            if self.dist is None or self.dist.is_main_process:
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
                if getattr(self.model, "aux_dim", 0):
                    outputs = self.model(coords, params, sdf, aux=aux)
                else:
                    outputs = self.model(coords, params, sdf)
                loss = self.criterion(outputs, targets)
                bs = targets.size(0)
                total_loss += loss.item() * bs
                total_samples += bs
            stats = torch.tensor([total_loss, total_samples], device=self.device, dtype=torch.float64)
            if self.dist is not None and self.dist.enabled:
                self.dist.all_reduce_sum(stats)
        return (stats[0] / stats[1]).item()


    def save_model(self, path=None, low_fi=False):
        # Only rank 0 saves
        if self.dist is not None and not self.dist.is_main_process:
            return

        load_best_weights(self, f'Outputs/{self.project_name}/checkpoints/best_model.pt')

        if path is None:
            os.makedirs(f'Outputs/{self.project_name}/model', exist_ok=True)
            if low_fi:
                path = f'Outputs/{self.project_name}/model/low_fi_fusion_deeponet.pt'
            else:
                path = f'Outputs/{self.project_name}/model/fusion_deeponet.pt'

        state_dict = self._get_model_state_dict()
        torch.save(state_dict, path)




    def _get_model_state_dict(self):
        if self.dist is not None and self.dist.enabled:
            return FSDP.state_dict(self.model)
        else:
            return self.model.state_dict()

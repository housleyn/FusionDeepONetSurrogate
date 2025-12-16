import torch
import os
from src.utils.distributed import is_main_process
import time

class MethodsTrainer:
    def train(self, train_loader, test_loader, num_epochs, print_every):
        loss_history, test_loss_history = [], []
        epoch_times = []
        best_loss = float('inf')
        if is_main_process():
            os.makedirs(f'Outputs/{self.project_name}/checkpoints', exist_ok=True)

        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            self.model.train()
            total_samples = 0

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            for coords, params, targets, sdf, *maybe_aux in train_loader:
                coords, params, targets, sdf, aux = self._move_batch_to_device(
                    coords, params, targets, sdf, maybe_aux[0] if len(maybe_aux) else None
                )
                
                coords.requires_grad = True
               
                params.requires_grad = True
                
                targets.requires_grad = True
                
                

                self.optimizer.zero_grad()
                outputs = self.model(coords, params, sdf, aux=aux)
                
                
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                self.optimizer.step()

                batch_size = targets.size(0)
                epoch_loss += loss.item() * batch_size
                total_samples += batch_size

            if (epoch + 1) % 10000 == 0:
                 self.lr_scheduler.step()
            avg_loss = epoch_loss / total_samples
            loss_history.append(avg_loss)

            # Evaluate on the test set
            test_loss = self.evaluate(test_loader)
            test_loss_history.append(test_loss)

            # Checkpoint: Save best model
            if test_loss < best_loss:
                best_loss = test_loss
                if is_main_process():
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.lr_scheduler.state_dict(),
                        'loss': test_loss
                    }
                    torch.save(checkpoint, f'Outputs/{self.project_name}/checkpoints/best_model.pt')

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            if is_main_process() and (epoch % print_every == 0 or epoch == num_epochs - 1):
                    print(f"Epoch {epoch:4d} | Train Loss: {avg_loss:.6f} | Test Loss: {test_loss:.6f} | LR: {self.lr_scheduler.get_last_lr()[0]:.2e} | epoch Time: {epoch_time:.6f}s | Avg epoch Time: {sum(epoch_times)/len(epoch_times):.6f}s")

        return loss_history, test_loss_history

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for coords, params, targets, sdf, *maybe_aux in dataloader:
                coords, params, targets, sdf, aux = self._move_batch_to_device(
                    coords, params, targets, sdf, maybe_aux[0] if len(maybe_aux) else None
                )
                

                outputs = self.model(coords, params, sdf, aux=aux)
                if self.loss_type == "mse":
                    loss = self.criterion(outputs, targets)
                # elif self.loss_type == "weighted_mse":
                #     loss = self.weighted_mse(outputs, targets)
                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        return total_loss / total_samples

    def save_model(self, path=None, low_fi=False):
        if not is_main_process():
            return
        if path is None:
            os.makedirs(f'Outputs/{self.project_name}/model', exist_ok=True)
            if low_fi:
                path = f'Outputs/{self.project_name}/model/low_fi_fusion_deeponet.pt'
            else:
                path = f'Outputs/{self.project_name}/model/fusion_deeponet.pt'
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

        self.model.eval()

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        return start_epoch

    def weighted_mse(self, pred, target, weights):
        weights = weights.unsqueeze(0).unsqueeze(-1)
        return ((pred-target) ** 2 * weights).mean()
    
    def _model_device(self):
        return next(self.model.parameters()).device

    def _move_batch_to_device(self, coords, params, targets, sdf, aux=None):
        model_device = self._model_device()
        non_blocking = model_device.type == "cuda"
        coords = coords.to(model_device, non_blocking=non_blocking)
        params = params.to(model_device, non_blocking=non_blocking)
        targets = targets.to(model_device, non_blocking=non_blocking)
        sdf = sdf.to(model_device, non_blocking=non_blocking)
        aux = aux.to(model_device, non_blocking=non_blocking) if aux is not None else None

        assert coords.device == model_device, f"coords device {coords.device} != model device {model_device}"
        assert params.device == model_device, f"params device {params.device} != model device {model_device}"
        assert targets.device == model_device, f"targets device {targets.device} != model device {model_device}"
        assert sdf.device == model_device, f"sdf device {sdf.device} != model device {model_device}"
        if aux is not None:
            assert aux.device == model_device, f"aux device {aux.device} != model device {model_device}"

        return coords, params, targets, sdf, aux
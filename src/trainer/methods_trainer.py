import os
import time
import statistics
import torch
from src.utils.distributed import is_main_process


class MethodsTrainer:
    def train(self, train_loader, test_loader, num_epochs, print_every):
        loss_history, test_loss_history = [], []
        epoch_times = []
        best_loss = float("inf")
        if is_main_process():
            os.makedirs(f"Outputs/{self.project_name}/checkpoints", exist_ok=True)

        for epoch in range(num_epochs):
            self._assert_model_single_device()

            epoch_start = time.time()
            epoch_loss = 0.0
            self.model.train()
            total_samples = 0

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            data_times = []
            forward_times = []
            backward_times = []
            optim_times = []
            step_times = []

            for coords, params, targets, sdf, *maybe_aux in train_loader:
                batch_start = time.perf_counter()
                coords, params, targets, sdf, aux = self._move_batch_to_device(
                    coords, params, targets, sdf, maybe_aux[0] if len(maybe_aux) else None
                )

                data_ready = time.perf_counter()
                data_times.append(data_ready - batch_start)

                coords.requires_grad = True
                params.requires_grad = True
                targets.requires_grad = True

                self.optimizer.zero_grad()
                fwd_start = time.perf_counter()
                outputs = self.model(coords, params, sdf, aux=aux)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                forward_times.append(time.perf_counter() - fwd_start)

                loss = self.criterion(outputs, targets)
                bw_start = time.perf_counter()
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                backward_times.append(time.perf_counter() - bw_start)

                opt_start = time.perf_counter()
                self.optimizer.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                optim_times.append(time.perf_counter() - opt_start)

                step_times.append(time.perf_counter() - batch_start)

                batch_size = targets.size(0)
                epoch_loss += loss.item() * batch_size
                total_samples += batch_size

            if (epoch + 1) % 10000 == 0:
                self.lr_scheduler.step()
            avg_loss = epoch_loss / max(1, total_samples)
            loss_history.append(avg_loss)

            test_loss = self.evaluate(test_loader)
            test_loss_history.append(test_loss)

            if test_loss < best_loss:
                best_loss = test_loss
                if is_main_process():
                    m = self.model.module if hasattr(self.model, "module") else self.model
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": m.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.lr_scheduler.state_dict(),
                        "loss": test_loss,
                    }
                    torch.save(checkpoint, f"Outputs/{self.project_name}/checkpoints/best_model.pt")

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            self.last_epoch_timing = {
                "data_time_mean": statistics.mean(data_times) if data_times else 0.0,
                "data_time_p95": percentile(data_times, 95) if data_times else 0.0,
                "forward_mean": statistics.mean(forward_times) if forward_times else 0.0,
                "backward_mean": statistics.mean(backward_times) if backward_times else 0.0,
                "optim_mean": statistics.mean(optim_times) if optim_times else 0.0,
                "step_mean": statistics.mean(step_times) if step_times else 0.0,
            }

            if is_main_process() and (epoch % print_every == 0 or epoch == num_epochs - 1):
                print(
                    f"Epoch {epoch:4d} | Train Loss: {avg_loss:.6f} | Test Loss: {test_loss:.6f} | LR: {self.lr_scheduler.get_last_lr()[0]:.2e} | "
                    f"epoch Time: {epoch_time:.6f}s | Avg epoch Time: {sum(epoch_times)/len(epoch_times):.6f}s | "
                    f"data mean {self.last_epoch_timing['data_time_mean']:.4f}s | fwd {self.last_epoch_timing['forward_mean']:.4f}s | "
                    f"bwd {self.last_epoch_timing['backward_mean']:.4f}s | opt {self.last_epoch_timing['optim_mean']:.4f}s"
                )

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
                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        return total_loss / max(1, total_samples)

    def save_model(self, path=None, low_fi=False):
        if not is_main_process():
            return
        if path is None:
            os.makedirs(f"Outputs/{self.project_name}/model", exist_ok=True)
            path = f"Outputs/{self.project_name}/model/low_fi_fusion_deeponet.pt" if low_fi else f"Outputs/{self.project_name}/model/fusion_deeponet.pt"
        m = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(m.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        return start_epoch

    def weighted_mse(self, pred, target, weights):
        weights = weights.unsqueeze(0).unsqueeze(-1)
        return ((pred - target) ** 2 * weights).mean()

    def _model_device(self):
        return self._assert_model_single_device()


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
    
    def _assert_model_single_device(self):
            # unwrap DDP if needed
            m = self.model.module if hasattr(self.model, "module") else self.model

            devs = {p.device for p in m.parameters()}
            buf_devs = {b.device for b in m.buffers()}
            all_devs = devs | buf_devs

            assert len(all_devs) == 1, (
                f"Model is split across devices! param_devs={sorted(map(str, devs))} "
                f"buffer_devs={sorted(map(str, buf_devs))}"
            )
            return next(iter(all_devs))

def percentile(values, q):
        if not values:
            return 0.0
        k = int(len(values) * q / 100)
        return sorted(values)[min(k, len(values) - 1)]
    
    

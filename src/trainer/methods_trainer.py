import torch
import os

class MethodsTrainer:
    def train(self, train_loader, test_loader, num_epochs, print_every):
        loss_history, test_loss_history = [], []
        best_loss = float('inf')
        os.makedirs(f'Outputs/{self.project_name}/checkpoints', exist_ok=True)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            self.model.train()
            total_samples = 0

            for coords, params, targets, sdf in train_loader:
                coords = coords.to(self.device)
                params = params.to(self.device)
                targets = targets.to(self.device)
                sdf = sdf.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(coords, params, sdf)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                batch_size = targets.size(0)
                epoch_loss += loss.item() * batch_size
                total_samples += batch_size

            # if (epoch + 1) % 20000 == 0:
            #      self.lr_scheduler.step()
            avg_loss = epoch_loss / total_samples
            loss_history.append(avg_loss)

            # Evaluate on the test set
            test_loss = self.evaluate(test_loader)
            test_loss_history.append(test_loss)

            # Checkpoint: Save best model
            if test_loss < best_loss:
                best_loss = test_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'loss': test_loss
                }
                torch.save(checkpoint, f'Outputs/{self.project_name}/checkpoints/best_model.pt')

            if epoch % print_every == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:4d} | Train Loss: {avg_loss:.6f} | Test Loss: {test_loss:.6f} | LR: {self.lr_scheduler.get_last_lr()[0]:.2e}")

        return loss_history, test_loss_history

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for coords, params, targets, sdf in dataloader:
                coords = coords.to(self.device)
                params = params.to(self.device)
                targets = targets.to(self.device)
                sdf = sdf.to(self.device)

                outputs = self.model(coords, params, sdf)
                loss = self.criterion(outputs, targets)
                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        return total_loss / total_samples

    def save_model(self):
        os.makedirs(f'Outputs/{self.project_name}/model', exist_ok=True)
        path = f'Outputs/{self.project_name}/model/fusion_deeponet.pt'
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        return start_epoch

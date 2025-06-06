import torch 
class MethodsTrainer:
    def train(self, train_loader, test_loader, num_epochs=1000, print_every=100):
        self.model.train()
        loss_history, test_loss_history = [], []
        for coords, params, targets in train_loader:
            print("coords shape:", coords.shape)
            print("params shape:", params.shape)
            print("targets shape:", targets.shape)
            break

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for coords, params, targets in train_loader:
                coords = coords.to(self.device)
                params = params.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(coords, params)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            self.lr_scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)

            # Evaluate on the test set
            test_loss = self.evaluate(test_loader)
            test_loss_history.append(test_loss)

            if epoch % print_every == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:4d} | Train Loss: {avg_loss:.6f} | Test Loss: {test_loss:.6f} | LR: {self.lr_scheduler.get_last_lr()[0]:.2e}")

        return loss_history, test_loss_history

    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for coords, params, targets in dataloader:
                coords = coords.to(self.device)
                params = params.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(coords, params)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss


    def save_model(self, path="fusion_deeponet.pt"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
import torch 
class MethodsTrainer:
    def train(self, train_loader, test_loader, num_epochs, print_every):
        
        loss_history, test_loss_history = [], []
        # for coords, params, targets in train_loader:
        #     # Print shapes
        #     print("coords shape:", coords.shape)
        #     print("params shape:", params.shape)
        #     print("targets shape:", targets.shape)

        #     # Print column headings and first row for coords
        #     print("coords columns: ['X', 'Y', 'Z'] (example)")
        #     print("coords first row:", coords[0, 0].cpu().numpy())

        #     # Print column headings and first row for params
        #     print("params columns: ['radius'] (example)")
        #     print("params first row:", params[0].cpu().numpy())

        #     # Print column headings and first row for targets
        #     print("targets columns: ['u', 'v', 'w', 'rho', 'p'] (example)")
        #     print("targets first row:", targets[0, 0].cpu().numpy())

        #     break  # Remove this break if you want to print for every batch

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            self.model.train()
            total_samples = 0

            for coords, params, targets in train_loader:
                coords = coords.to(self.device)
                params = params.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(coords, params)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                batch_size = targets.size(0)
                epoch_loss += loss.item() * batch_size
                total_samples += batch_size


            # self.lr_scheduler.step()
            avg_loss = epoch_loss / total_samples
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
        total_samples = 0
        with torch.no_grad():
            for coords, params, targets in dataloader:
                coords = coords.to(self.device)
                params = params.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(coords, params)
                loss = self.criterion(outputs, targets)
                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        return avg_loss


    def save_model(self, path="fusion_deeponet.pt"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MethodsDataloader:
    def __len__(self):
        return self.coords.shape[0]  # number of samples

    def __getitem__(self, idx):
        return self.coords[idx], self.params[idx], self.outputs[idx]



    def get_dataloader(self, batch_size, shuffle=True, test_size=.25):
        coords = self.coords
        outputs = self.outputs
        params = self.params

        # Split data into training and testing sets
        indices = np.arange(coords.shape[0])
        train_indices, test_indices = train_test_split(indices, test_size=test_size, shuffle=shuffle, random_state=42)

        train_dataset = torch.utils.data.TensorDataset(
            coords[train_indices], params[train_indices], outputs[train_indices]
        )
        test_dataset = torch.utils.data.TensorDataset(
            coords[test_indices], params[test_indices], outputs[test_indices]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
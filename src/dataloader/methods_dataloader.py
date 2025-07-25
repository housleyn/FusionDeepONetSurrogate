import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MethodsDataloader:

    def get_dataloader(self, batch_size, shuffle=True, test_size=.2):
        coords = self.coords
        outputs = self.outputs
        params = self.params
        sdf = self.sdf

        indices = np.arange(coords.shape[0])
        train_indices, test_indices = train_test_split(indices, test_size=test_size, shuffle=shuffle, random_state=42)

        train_dataset = torch.utils.data.TensorDataset(
            coords[train_indices], params[train_indices], outputs[train_indices], sdf[train_indices]
        )
        test_dataset = torch.utils.data.TensorDataset(
            coords[test_indices], params[test_indices], outputs[test_indices], sdf[test_indices]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
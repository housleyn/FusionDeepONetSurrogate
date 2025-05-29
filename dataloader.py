import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class FusionCFDDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.coords = torch.tensor(data["coords"], dtype=torch.float32)    # (B, N, 3)
        self.outputs = torch.tensor(data["outputs"], dtype=torch.float32)  # (B, N, 4/5)
        self.params = torch.tensor(data["params"], dtype=torch.float32)    # (B, 1)

    def __len__(self):
        return self.coords.shape[0]  # number of samples

    def __getitem__(self, idx):
        return self.coords[idx], self.params[idx], self.outputs[idx]



def get_dataloader(npz_path, batch_size=1, shuffle=True, test_size=2/9):
    data = np.load(npz_path)
    coords = torch.tensor(data["coords"], dtype=torch.float32)
    outputs = torch.tensor(data["outputs"], dtype=torch.float32)
    params = torch.tensor(data["params"], dtype=torch.float32)

    # Split data into training and testing sets
    indices = np.arange(coords.shape[0])
    train_indices, test_indices = train_test_split(indices, test_size=test_size, shuffle=shuffle, random_state=42)

    train_dataset = torch.utils.data.TensorDataset(coords[train_indices], params[train_indices], outputs[train_indices])
    test_dataset = torch.utils.data.TensorDataset(coords[test_indices], params[test_indices], outputs[test_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


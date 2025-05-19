import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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

def get_dataloader(npz_path, batch_size=1, shuffle=True):
    dataset = FusionCFDDataset(npz_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

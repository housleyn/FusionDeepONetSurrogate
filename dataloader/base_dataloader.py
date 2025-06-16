import torch
import numpy as np
class BaseDataloader:
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.coords = torch.tensor(data["coords"], dtype=torch.float32)    # (B, N, 3)
        self.outputs = torch.tensor(data["outputs"], dtype=torch.float32)  # (B, N, 4/5)
        self.params = torch.tensor(data["params"], dtype=torch.float32)    # (B, 1)

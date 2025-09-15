import torch
import numpy as np
class BaseDataloader:
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.coords = torch.tensor(data["coords"], dtype=torch.float32)    # (B, N, 3)
        self.outputs = torch.tensor(data["outputs"], dtype=torch.float32)  # (B, N, 6)
        self.params = torch.tensor(data["params"], dtype=torch.float32)    # (B, 1)
        self.sdf = torch.tensor(data["sdf"], dtype=torch.float32)          # (B, N, 1)
        # self.weights = torch.tensor(data["weights"], dtype=torch.float32)  # (B, N, 1)
        

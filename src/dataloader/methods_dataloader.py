import torch 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class MethodsDataloader:

    def get_dataloader(self, batch_size, shuffle=True, test_size=.2):

        idx = np.arange(self.coords.shape[0])
        tri, tei = train_test_split(idx, test_size=test_size, shuffle=shuffle, random_state=42)
        train_loader = DataLoader(self.make_ds(tri), batch_size=batch_size, shuffle=shuffle)
        test_loader  = DataLoader(self.make_ds(tei), batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    
    def make_ds(self, I):
            
            if self.aux_lf is None:
                return TensorDataset(self.coords[I], self.params[I], self.outputs[I], self.sdf[I])
            else:
                return TensorDataset(self.coords[I], self.params[I], self.outputs[I], self.sdf[I], self.aux_lf[I])
import torch 
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class MethodsDataloader:

    def get_dataloader(self, batch_size, shuffle=True, test_size=.2):
        coords, outputs, params, sdf = self.coords, self.outputs, self.params, self.sdf
        aux = self.aux_lf  if hasattr(self, "aux_lf") else None

        idx = np.arange(coords.shape[0])
        tri, tei = train_test_split(idx, test_size=test_size, shuffle=shuffle, random_state=42)

        def make_ds(I):
            if aux is None:
                return TensorDataset(coords[I], params[I], outputs[I], sdf[I])
            else:
                return TensorDataset(coords[I], params[I], outputs[I], sdf[I], aux[I])

        train_loader = DataLoader(make_ds(tri), batch_size=batch_size, shuffle=shuffle)
        test_loader  = DataLoader(make_ds(tei), batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
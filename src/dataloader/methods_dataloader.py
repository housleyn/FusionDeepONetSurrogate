import torch 
import torch.distributed as dist
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split

class MethodsDataloader:

    def get_dataloader(self, batch_size, shuffle=True, test_size=.2, ddp=False, world_size=1, rank=0):
        coords, outputs, params, sdf = self.coords, self.outputs, self.params, self.sdf
        aux = self.aux_lf  if hasattr(self, "aux_lf") else None

        idx = np.arange(coords.shape[0])
        tri, tei = train_test_split(idx, test_size=test_size, shuffle=shuffle, random_state=42)

        def make_ds(I):
            if aux is None:
                return TensorDataset(coords[I], params[I], outputs[I], sdf[I])
            else:
                return TensorDataset(coords[I], params[I], outputs[I], sdf[I], aux[I])

        

        train_sampler = None
        test_sampler = None 

        if ddp and dist.is_available() and dist.is_initialized():
            train_dataset = make_ds(tri)
            test_dataset = make_ds(tei)
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
            test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False)
        else:
            train_loader = DataLoader(make_ds(tri), batch_size=batch_size, shuffle=shuffle)
            test_loader = DataLoader(make_ds(tei), batch_size=batch_size, shuffle=False)


        return train_loader, test_loader, train_sampler
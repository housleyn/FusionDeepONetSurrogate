import torch 
import torch.distributed as dist
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
import os

def get_default_num_workers():
    cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus_per_task is not None:
        try:
            n = int(cpus_per_task)
            return max(1, n - 1)
        except ValueError:
            pass
    n_local = os.cpu_count() or 1
    return max(1, n_local - 1)

class MethodsDataloader:

    def get_dataloader(self, batch_size, shuffle=True, test_size=.2, ddp=False, world_size=1, rank=0, num_workers=None):
        coords, outputs, params, sdf = self.coords, self.outputs, self.params, self.sdf
        aux = self.aux_lf  if hasattr(self, "aux_lf") else None

        if num_workers is None:
            num_workers = 0 #get_default_num_workers()

        pin_memory = torch.cuda.is_available()
        persistent = num_workers > 0

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

            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
        else:
            train_loader = DataLoader(make_ds(tri), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
            test_loader = DataLoader(make_ds(tei), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)

        if rank == 0:
            print(f"[DDP] num_workers = {num_workers}, pin_memory = {pin_memory}")

        return train_loader, test_loader, train_sampler
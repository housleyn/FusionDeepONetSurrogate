import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split


class MethodsDataloader:

    def get_dataloader(self, batch_size, shuffle=True, test_size=0.2, dist_context=None):
        idx = np.arange(self.coords.shape[0])
        tri, tei = train_test_split(
            idx,
            test_size=test_size,
            shuffle=shuffle,
            random_state=42
        )

        train_ds = self.make_ds(tri)
        test_ds = self.make_ds(tei)

        train_sampler = None
        test_sampler = None

        if dist_context is not None and dist_context.enabled:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=dist_context.world_size,
                rank=dist_context.rank,
                shuffle=shuffle
            )
            test_sampler = DistributedSampler(
                test_ds,
                num_replicas=dist_context.world_size,
                rank=dist_context.rank,
                shuffle=False
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(train_sampler is None and shuffle),
            sampler=train_sampler
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=test_sampler
        )

        return train_loader, test_loader, train_sampler, test_sampler

    def make_ds(self, I):
        if self.aux_lf is None:
            return TensorDataset(self.coords[I], self.params[I], self.outputs[I], self.sdf[I])
        else:
            return TensorDataset(self.coords[I], self.params[I], self.outputs[I], self.sdf[I], self.aux_lf[I])
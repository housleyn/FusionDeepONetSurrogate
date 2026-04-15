import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class MethodsDataloader:
    def get_dataloader(
        self,
        batch_size,
        shuffle=True,
        test_size=0.2,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    ):
        idx = np.arange(self.coords.shape[0])
        tri, tei = train_test_split(
            idx,
            test_size=test_size,
            shuffle=shuffle,
            random_state=42,
        )

        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory and torch.cuda.is_available(),
            "persistent_workers": persistent_workers and num_workers > 0,
        }

        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor

        train_loader = DataLoader(
            self.make_ds(tri),
            shuffle=shuffle,
            **loader_kwargs,
        )

        test_loader = DataLoader(
            self.make_ds(tei),
            shuffle=False,
            **loader_kwargs,
        )

        return train_loader, test_loader

    def make_ds(self, I):
        if self.aux_lf is None:
            return TensorDataset(
                self.coords[I],
                self.params[I],
                self.outputs[I],
                self.sdf[I],
            )
        else:
            return TensorDataset(
                self.coords[I],
                self.params[I],
                self.outputs[I],
                self.sdf[I],
                self.aux_lf[I],
            )
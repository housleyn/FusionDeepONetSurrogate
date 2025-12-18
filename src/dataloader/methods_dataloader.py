import os
import numpy as np
import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


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
    def get_dataloader(self, batch_size, shuffle=True, test_size=0.2, ddp=False, num_workers=None):
        coords, outputs, params, sdf = self.coords, self.outputs, self.params, self.sdf
        aux = self.aux_lf if hasattr(self, "aux_lf") else None

        if num_workers is None:
            num_workers = 0  # or get_default_num_workers()

        ddp_active = ddp and dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if ddp_active else 0
        world_size = dist.get_world_size() if ddp_active else 1

        pin_memory = torch.cuda.is_available()
        persistent = num_workers > 0

        idx = np.arange(coords.shape[0])

        # -------------------------------
        # Make train/test split ONCE (rank0) and broadcast to all ranks
        # -------------------------------
        if ddp_active:
            dev = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")

            if rank == 0:
                tri_np, tei_np = train_test_split(
                    idx, test_size=test_size, shuffle=shuffle, random_state=42
                )
                tri_t = torch.tensor(tri_np, dtype=torch.long, device=dev)
                tei_t = torch.tensor(tei_np, dtype=torch.long, device=dev)
                lens = torch.tensor([tri_t.numel(), tei_t.numel()], dtype=torch.long, device=dev)
            else:
                tri_t = torch.empty(1, dtype=torch.long, device=dev)  # placeholder
                tei_t = torch.empty(1, dtype=torch.long, device=dev)  # placeholder
                lens = torch.empty(2, dtype=torch.long, device=dev)

            dist.broadcast(lens, src=0)
            tri_len, tei_len = int(lens[0].item()), int(lens[1].item())

            if rank != 0:
                tri_t = torch.empty(tri_len, dtype=torch.long, device=dev)
                tei_t = torch.empty(tei_len, dtype=torch.long, device=dev)

            dist.broadcast(tri_t, src=0)
            dist.broadcast(tei_t, src=0)

            tri = tri_t.cpu().numpy()
            tei = tei_t.cpu().numpy()

        else:
            tri, tei = train_test_split(
                idx, test_size=test_size, shuffle=shuffle, random_state=42
            )

        # -------------------------------
        # Dataset construction
        # -------------------------------
        def make_ds(indices):
            if aux is None:
                return TensorDataset(coords[indices], params[indices], outputs[indices], sdf[indices])
            return TensorDataset(coords[indices], params[indices], outputs[indices], sdf[indices], aux[indices])

        train_dataset = make_ds(tri)
        test_dataset = make_ds(tei)

        train_sampler = None

        # -------------------------------
        # DataLoaders
        # -------------------------------
        if ddp_active:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=False,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent,
            )

            # Keep evaluation simple + safe with your current Trainer.evaluate():
            # Rank0 evaluates real test set, other ranks get an empty loader.
            if rank == 0:
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=persistent,
                )
            else:
                empty = TensorDataset(
                    coords[:0], params[:0], outputs[:0], sdf[:0], *( [aux[:0]] if aux is not None else [] )
                )
                test_loader = DataLoader(
                    empty,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    persistent_workers=False,
                )

        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent,
            )

        # -------------------------------
        # Debug prints
        # -------------------------------
        if (not ddp_active) or rank == 0:
            print(f"[DL DEBUG] ddp_active={ddp_active} rank={rank}/{world_size}")
            print(f"[DL DEBUG] sampler={type(train_loader.sampler)}")
            try:
                print(
                    f"[DL DEBUG] len(dataset)={len(train_loader.dataset)} "
                    f"len(sampler)={len(train_loader.sampler)} len(loader)={len(train_loader)}"
                )
            except Exception as e:
                print(f"[DL DEBUG] len() error: {e}")

        if rank == 0:
            print(f"[DDP] num_workers = {num_workers}, pin_memory = {pin_memory}, persistent_workers = {persistent}")

        return train_loader, test_loader, train_sampler

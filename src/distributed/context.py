import os
import torch
import torch.distributed as dist


class DistributedContext:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self):
        if not self.enabled:
            return

        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training with NCCL requires CUDA.")

        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available.")

        dist.init_process_group(backend="nccl")

        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")

    def cleanup(self):
        if self.enabled and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    @property
    def is_main_process(self):
        return self.rank == 0
    
    def barrier(self):
        if self.enabled and dist.is_available() and dist.is_initialized():
            dist.barrier()

    def all_reduce_sum(self, value_tensor):
        if self.enabled and dist.is_available() and dist.is_initialized():
            dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
        return value_tensor
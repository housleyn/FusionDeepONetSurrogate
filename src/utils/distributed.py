import os
import torch
import torch.distributed as dist


def setup_ddp():
    """Initialize DDP if environment variables are set.

    Returns a dictionary with keys:
        - is_ddp (bool): whether distributed is initialized
        - rank (int): global rank (defaults to 0 when not distributed)
        - local_rank (int): local rank on the current node (defaults to 0)
        - world_size (int): total number of processes
        - device (torch.device): device assigned to this process
    """
    required_env = ("LOCAL_RANK", "RANK", "WORLD_SIZE")
    has_env = all(k in os.environ for k in required_env)

    if not has_env:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {"is_ddp": False, "rank": 0, "local_rank": 0, "world_size": 1, "device": device}

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not torch.cuda.is_available():
        device = torch.device("cpu")
        return {"is_ddp": False, "rank": 0, "local_rank": 0, "world_size": 1, "device": device}

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return {"is_ddp": True, "rank": rank, "local_rank": local_rank, "world_size": world_size, "device": device}


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0
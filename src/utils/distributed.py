import os
import torch
import torch.distributed as dist


# NCCL/Gloo friendly helper defaults
_DEFAULT_BACKEND = "nccl" if torch.cuda.is_available() else "gloo"


def setup_ddp():
    """Initialize torch.distributed if launch env vars are present.

    Returns:
        dict: keys: is_ddp, rank, local_rank, world_size, device
    """
    required_env = ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    has_env = all(k in os.environ for k in required_env)

    if not has_env:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {"is_ddp": False, "rank": 0, "local_rank": 0, "world_size": 1, "device": device}

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[rank {rank}] local_rank={local_rank} current_device={torch.cuda.current_device()} "
          f"visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")


    if not dist.is_initialized():
        if torch.cuda.is_available():
            print(f"[rank {rank}] local_rank={local_rank} current_device={torch.cuda.current_device()} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")

        dist.init_process_group(
            backend=_DEFAULT_BACKEND,
            init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None,

        )


    return {"is_ddp": True, "rank": rank, "local_rank": local_rank, "world_size": world_size, "device": device}


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def ddp_info_summary():
    """Return a human readable summary of the distributed environment."""
    if not dist.is_initialized():
        return "DDP disabled"
    backend = dist.get_backend()
    return f"DDP backend={backend}, rank={dist.get_rank()}, world_size={dist.get_world_size()}"

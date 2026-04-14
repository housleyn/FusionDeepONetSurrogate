import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def wrap_model_for_fsdp(model, dist_context):
    if not dist_context.enabled:
        return model

    if not torch.cuda.is_available():
        raise RuntimeError("FSDP requires CUDA.")

    model = model.to(dist_context.device)

    device_id = dist_context.device.index
    model = FSDP(model, device_id=device_id)

    return model
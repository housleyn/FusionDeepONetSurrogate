import argparse
import os
import socket
import statistics
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from src.utils.distributed import barrier, ddp_info_summary, is_main_process, setup_ddp


def _find_free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def build_synthetic_dataset(num_samples: int = 128, feature_dim: int = 8):
    coords = torch.randn(num_samples, feature_dim)
    params = torch.randn(num_samples, feature_dim)
    targets = torch.randn(num_samples, feature_dim)
    sdf = torch.randn(num_samples, feature_dim)
    return TensorDataset(coords, params, targets, sdf)


def make_dataloader(dataset, batch_size: int, ddp_info: Dict, shuffle: bool = True, num_workers: int = 0):
    sampler = None
    if ddp_info.get("is_ddp"):
        sampler = DistributedSampler(
            dataset,
            num_replicas=ddp_info.get("world_size", 1),
            rank=ddp_info.get("rank", 0),
            shuffle=shuffle,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def benchmark_dataloader(loader: DataLoader, steps: int = 20) -> Dict[str, float]:
    latencies: List[float] = []
    start_time = time.perf_counter()
    for i, batch in enumerate(loader):
        batch_start = time.perf_counter()
        _ = batch
        latencies.append(time.perf_counter() - batch_start)
        if i + 1 >= steps:
            break
    total_time = time.perf_counter() - start_time
    batches_seen = len(latencies)
    return {
        "batches": batches_seen,
        "batches_per_sec": batches_seen / total_time if total_time > 0 else 0.0,
        "mean_load_time": statistics.mean(latencies) if latencies else 0.0,
        "p95_load_time": percentile(latencies, 95) if latencies else 0.0,
    }


def benchmark_training_step(loader: DataLoader, device: torch.device, ddp_info: Dict, steps: int = 10) -> Dict[str, float]:
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    if ddp_info.get("is_ddp"):
        ddp_kwargs = {
            "device_ids": [ddp_info.get("local_rank", 0)] if torch.cuda.is_available() else None,
            "output_device": ddp_info.get("local_rank", 0) if torch.cuda.is_available() else None,
            "find_unused_parameters": False,
            "gradient_as_bucket_view": True,
        }
        if not torch.cuda.is_available():
            ddp_kwargs.pop("device_ids")
            ddp_kwargs.pop("output_device")
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)

    forward_times: List[float] = []
    backward_times: List[float] = []
    no_sync_backward_times: List[float] = []
    optim_times: List[float] = []

    for i, batch in enumerate(loader):
        coords, params, targets, sdf = [x.to(device, non_blocking=True) for x in batch]
        optimizer.zero_grad()

        fwd_start = time.perf_counter()
        outputs = model(coords)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_times.append(time.perf_counter() - fwd_start)

        loss = loss_fn(outputs, targets)
        bwd_start = time.perf_counter()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        backward_times.append(time.perf_counter() - bwd_start)

        if ddp_info.get("is_ddp"):
            with model.no_sync():
                optimizer.zero_grad()
                outputs = model(coords)
                loss = loss_fn(outputs, targets)
                bwd_ns_start = time.perf_counter()
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                no_sync_backward_times.append(time.perf_counter() - bwd_ns_start)

        opt_start = time.perf_counter()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        optim_times.append(time.perf_counter() - opt_start)

        if i + 1 >= steps:
            break

    mean_forward = statistics.mean(forward_times) if forward_times else 0.0
    mean_backward = statistics.mean(backward_times) if backward_times else 0.0
    mean_backward_no_sync = statistics.mean(no_sync_backward_times) if no_sync_backward_times else 0.0
    mean_opt = statistics.mean(optim_times) if optim_times else 0.0

    comm_overhead = max(0.0, mean_backward - mean_backward_no_sync) if no_sync_backward_times else 0.0

    return {
        "forward_mean": mean_forward,
        "backward_mean": mean_backward,
        "backward_no_sync_mean": mean_backward_no_sync,
        "estimated_comm_time": comm_overhead,
        "optim_mean": mean_opt,
    }


def sampler_diagnostics(loader: DataLoader, ddp_info: Dict) -> Dict[str, str]:
    sampler = getattr(loader, "sampler", None)
    if not ddp_info.get("is_ddp"):
        return {"status": "DDP disabled", "detail": "single process run"}
    if sampler is None:
        return {"status": "error", "detail": "no sampler attached"}
    if not isinstance(sampler, DistributedSampler):
        return {"status": "error", "detail": f"expected DistributedSampler, got {type(sampler)}"}
    detail = f"DistributedSampler len={len(sampler)} shuffle={sampler.shuffle}"
    return {"status": "ok", "detail": detail}


def duplicate_work_check(block_name: str):
    if is_main_process():
        print(f"[rank0-only] executing {block_name}")
    else:
        print(f"[rank{dist.get_rank()}] skipped {block_name}")


def batch_size_report(config_batch: int, ddp_info: Dict) -> Dict[str, int]:
    world = ddp_info.get("world_size", 1)
    per_gpu = config_batch
    global_batch = per_gpu * world
    return {"per_rank_batch": per_gpu, "world_size": world, "global_batch": global_batch}


def log_nccl_env() -> Dict[str, str]:
    keys = [
        "NCCL_DEBUG",
        "NCCL_IB_DISABLE",
        "NCCL_P2P_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "NCCL_NET_GDR_LEVEL",
    ]
    return {k: os.environ.get(k, "<unset>") for k in keys}


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    k = int(len(values) * q / 100)
    return sorted(values)[min(k, len(values) - 1)]


def main():
    parser = argparse.ArgumentParser(description="DDP performance diagnostics")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    ddp_info = setup_ddp()
    if is_main_process():
        print(f"DDP info: {ddp_info_summary()}")

    dataset = build_synthetic_dataset()
    loader, sampler = make_dataloader(dataset, args.batch_size, ddp_info, shuffle=True, num_workers=args.num_workers)

    sampler_report = sampler_diagnostics(loader, ddp_info)
    data_report = benchmark_dataloader(loader, steps=args.steps)
    step_report = benchmark_training_step(loader, ddp_info.get("device", torch.device("cpu")), ddp_info, steps=args.steps)
    batch_report = batch_size_report(args.batch_size, ddp_info)
    env_report = log_nccl_env()

    if is_main_process():
        print("\n=== Diagnostics Summary ===")
        print(f"Sampler: {sampler_report}")
        print(f"Data loading: {data_report}")
        print(f"Step timing: {step_report}")
        print(f"Batch sizing: {batch_report}")
        print(f"NCCL env: {env_report}")

    barrier()


if __name__ == "__main__":
    main()

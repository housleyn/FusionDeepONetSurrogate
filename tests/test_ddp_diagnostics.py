import os
import sys
import tempfile
import pathlib

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.ddp_perf import batch_size_report, benchmark_dataloader, build_synthetic_dataset, duplicate_work_check, make_dataloader
from src.utils.distributed import cleanup_ddp, setup_ddp


def _run_sampler_worker(rank, world_size, port, q):
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dataset = torch.arange(16)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    sampler.set_epoch(0)
    indices = list(iter(sampler))
    q.put(indices)
    cleanup_ddp()


def test_sampler_unique_indices():
    world_size = 2
    port = 29555
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_run_sampler_worker, args=(rank, world_size, port, q)) for rank in range(world_size)]
    for p in procs:
        p.start()
    results = [q.get(timeout=10) for _ in range(world_size)]
    for p in procs:
        p.join()

    flat = [i for indices in results for i in indices]
    assert len(set(flat)) == len(flat) == 16


def _rank0_only_worker(rank, world_size, port, tmpdir):
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )
    ddp_info = setup_ddp()
    path = os.path.join(tmpdir, "rank0.txt")
    if ddp_info["rank"] == 0:
        duplicate_work_check("test")
        with open(path, "w") as f:
            f.write("ok")
    dist.barrier()
    cleanup_ddp()


def test_rank0_only_execution():
    world_size = 2
    port = 29565
    ctx = mp.get_context("spawn")
    tmpdir = tempfile.mkdtemp()
    procs = [ctx.Process(target=_rank0_only_worker, args=(rank, world_size, port, tmpdir)) for rank in range(world_size)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    assert os.path.exists(os.path.join(tmpdir, "rank0.txt"))
    files = [f for f in os.listdir(tmpdir) if f.endswith(".txt")]
    assert len(files) == 1


def test_dataloader_benchmark_smoke():
    ddp_info = {"is_ddp": False, "world_size": 1, "rank": 0}
    dataset = build_synthetic_dataset(num_samples=32, feature_dim=4)
    loader, _ = make_dataloader(dataset, batch_size=8, ddp_info=ddp_info, shuffle=True)
    report = benchmark_dataloader(loader, steps=5)
    assert report["batches"] > 0
    assert "mean_load_time" in report


def test_batch_size_report():
    report = batch_size_report(4, {"world_size": 2})
    assert report["global_batch"] == 8

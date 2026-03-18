import os
import yaml
from copy import deepcopy
from src.surrogate import Surrogate

BASE_YAML = "configs/x_43_sequential.yaml"

DATA_PAIRS = [
    ("Data/x_43_data_subset_18",  "Data/x_43_low_fi_data_subset_18"),
    ("Data/x_43_data_subset_144", "Data/x_43_low_fi_data_subset_144"),
]

OUT_DIR = "configs/sweeps/x_43_sequential"
UNSEEN_FOLDER = "Data/x_43_unseen_20"


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _save_yaml(path, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


def _subset_tag(folder: str) -> str:
    tail = os.path.basename(folder)
    return tail.split("_")[-1]  # e.g. "18", "72", "144", "200"


def _run_name(cfg):
    subset_tag = _subset_tag(cfg["data_folder"])
    return f"x_43_sequential_N{subset_tag}"


def build_run_specs():
    base_cfg = _load_yaml(BASE_YAML)
    specs = []

    for hf_folder, lf_folder in DATA_PAIRS:
        cfg = deepcopy(base_cfg)

        cfg["data_folder"] = hf_folder
        cfg["low_fi_data_folder"] = lf_folder

        name = _run_name(cfg)
        cfg["project_name"] = name

        cfg_path = os.path.join(OUT_DIR, f"{name}.yaml")
        specs.append((name, cfg_path, cfg))

    return specs


def run_one(cfg_path):
    s = Surrogate(config_path=cfg_path)
    s._train()
    if UNSEEN_FOLDER:
        s._infer_all_unseen(folder=UNSEEN_FOLDER)


if __name__ == "__main__":
    specs = build_run_specs()

    # Always rewrite configs so they match the current sweep
    for name, cfg_path, cfg in specs:
        _save_yaml(cfg_path, cfg)

    print(f"Generated {len(specs)} configs in: {OUT_DIR}")
    print("Example:", specs[0][1] if specs else "none")

    slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    # Local mode
    if slurm_id is None:
        for name, cfg_path, _ in specs:
            print(f"\n=== LOCAL === {name} -> {cfg_path}")
            run_one(cfg_path)
        raise SystemExit(0)

    # Slurm array mode
    idx = int(slurm_id)   # assumes --array=0-3
    if idx < 0 or idx >= len(specs):
        raise IndexError(
            f"SLURM_ARRAY_TASK_ID={idx} out of range (0..{len(specs)-1})"
        )

    name, cfg_path, _ = specs[idx]
    print(f"\n=== SLURM TASK {idx} === {name} -> {cfg_path}")
    run_one(cfg_path)
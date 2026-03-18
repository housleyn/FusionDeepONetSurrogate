import os
import yaml
from copy import deepcopy
from src.surrogate import Surrogate

BASE_YAML = "configs/waverider_multi.yaml"

HIDDEN_SIZES = [256, 512, 1024]

HF_FOLDERS = [
    "Data/waverider_hf_data_subset_50",
    "Data/waverider_hf_data_subset_100",
    "Data/waverider_hf_data_subset_200",
    "Data/waverider_hf_data_subset_300",
    "Data/waverider_hf_data_subset_400",
    "Data/waverider_hf_data_subset_500",
]

FIXED_LF_FOLDER = "Data/waverider_lf_data"

OUT_DIR = "configs/sweeps/waverider_multi"
UNSEEN_FOLDER = "Data/waverider_unseen"


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _save_yaml(path, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


def _subset_tag(folder: str) -> str:
    tail = os.path.basename(folder)
    parts = tail.split("_")
    return parts[-1]  # e.g. "500"


def _set_hidden_size(cfg, hidden_size):
    """
    Update the hidden-size field while preserving existing config structure
    as much as possible.
    """
    if "hidden_sizes" in cfg:
        if isinstance(cfg["hidden_sizes"], list):
            cfg["hidden_sizes"] = [hidden_size] * len(cfg["hidden_sizes"])
        else:
            cfg["hidden_sizes"] = hidden_size
    elif "hidden_size" in cfg:
        cfg["hidden_size"] = hidden_size
    else:
        cfg["hidden_sizes"] = hidden_size


def _get_hidden_tag(cfg):
    if "hidden_size" in cfg:
        return cfg["hidden_size"]
    if "hidden_sizes" in cfg:
        if isinstance(cfg["hidden_sizes"], list):
            return cfg["hidden_sizes"][0]
        return cfg["hidden_sizes"]
    return "unknown"


def _run_name(cfg, pair_mode):
    hf_tag = _subset_tag(cfg["data_folder"])
    lf_tag = _subset_tag(cfg["low_fi_data_folder"])
    hidden_tag = _get_hidden_tag(cfg)
    return f"waverider_{pair_mode}_HF{hf_tag}_LF{lf_tag}_H{hidden_tag}"


def build_folder_pairs():
    """
    Returns 12 folder pairs total:
    1. Fixed LF folder with each varying HF folder (6 pairs)
    2. Matching HF/HF folder pairs (6 pairs)
    """
    pairs = []

    # Case 1: same fixed LF folder with each HF folder
    for hf_folder in HF_FOLDERS:
        pairs.append(("multi", hf_folder, FIXED_LF_FOLDER))

    # Case 2: same HF folder repeated as LF folder
    for hf_folder in HF_FOLDERS:
        pairs.append(("sequential", hf_folder, hf_folder))

    return pairs


def build_run_specs():
    base_cfg = _load_yaml(BASE_YAML)
    specs = []

    model_type = base_cfg["model_type"]
    folder_pairs = build_folder_pairs()

    for pair_mode, hf_folder, lf_folder in folder_pairs:
        for hidden_size in HIDDEN_SIZES:
            cfg = deepcopy(base_cfg)

            cfg["model_type"] = model_type
            cfg["data_folder"] = hf_folder
            cfg["low_fi_data_folder"] = lf_folder
            _set_hidden_size(cfg, hidden_size)

            name = _run_name(cfg, pair_mode)
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
    idx = int(slurm_id)
    if idx < 0 or idx >= len(specs):
        raise IndexError(f"SLURM_ARRAY_TASK_ID={idx} out of range (0..{len(specs)-1})")

    name, cfg_path, _ = specs[idx]
    print(f"\n=== SLURM TASK {idx} === {name} -> {cfg_path}")
    run_one(cfg_path)
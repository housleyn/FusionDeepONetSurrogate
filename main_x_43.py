import os
import yaml
from copy import deepcopy
from itertools import product
from src.surrogate import Surrogate

BASE_YAML = "configs/x_43_transfer_learning_test.yaml"

DATA_PAIRS = [
    ("Data/x_43_data_36", "Data/x_43_low_fi_data_72"),  # multifidelity
    ("Data/x_43_data_36", "Data/x_43_data_36"),         # sequential
]

OUT_DIR = "configs/sweeps/x_43_transfer_ablation"
UNSEEN_FOLDER = "Data/x_43_unseen_20"

AUGMENTATION_OPTIONS = [False, True]
TRANSFER_OPTIONS = [False, True]


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _save_yaml(path, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


def _subset_tag(folder: str) -> str:
    tail = os.path.basename(folder)
    return tail.split("_")[-1]


def _bool_tag(value: bool, true_tag: str, false_tag: str) -> str:
    return true_tag if value else false_tag


def _mode_tag(hf_folder: str, lf_folder: str) -> str:
    return "seq" if hf_folder == lf_folder else "mf"


def _run_name(cfg):
    subset_tag = _subset_tag(cfg["data_folder"])
    mode_tag = _mode_tag(cfg["data_folder"], cfg["low_fi_data_folder"])
    aug_tag = _bool_tag(cfg["use_lf_augmentation"], "aug1", "aug0")
    tl_tag = _bool_tag(cfg["use_transfer_learning"], "tl1", "tl0")
    return f"x43_{mode_tag}_N{subset_tag}_{aug_tag}_{tl_tag}"


def build_run_specs():
    base_cfg = _load_yaml(BASE_YAML)
    specs = []

    for (hf_folder, lf_folder), use_aug, use_tl in product(
        DATA_PAIRS,
        AUGMENTATION_OPTIONS,
        TRANSFER_OPTIONS,
    ):
        cfg = deepcopy(base_cfg)
        cfg["data_folder"] = hf_folder
        cfg["low_fi_data_folder"] = lf_folder
        cfg["use_lf_augmentation"] = use_aug
        cfg["use_transfer_learning"] = use_tl

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

    for name, cfg_path, cfg in specs:
        _save_yaml(cfg_path, cfg)

    print(f"Generated {len(specs)} configs in: {OUT_DIR}")
    if specs:
        print("Example:", specs[0][1])

    slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if slurm_id is None:
        for i, (name, cfg_path, _) in enumerate(specs):
            print(f"\n=== LOCAL RUN {i}/{len(specs)-1} === {name} -> {cfg_path}")
            run_one(cfg_path)
        raise SystemExit(0)

    idx = int(slurm_id)
    if idx < 0 or idx >= len(specs):
        raise IndexError(f"SLURM_ARRAY_TASK_ID={idx} out of range (0..{len(specs)-1})")

    name, cfg_path, _ = specs[idx]
    print(f"\n=== SLURM TASK {idx} === {name} -> {cfg_path}")
    run_one(cfg_path)
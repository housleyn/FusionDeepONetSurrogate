# run_sweep.py
import os
import yaml
import itertools
from copy import deepcopy
from src.surrogate import Surrogate


# =========================
# 1) Sweep scope
# =========================
DATASETS = ["orion", "x43"]
MODEL_TYPES = ["sequential", "multi", "vanilla"]

BASE_YAMLS = {
    ("orion", "fusion"):       "configs/orion_fusion.yaml",
    # ("orion", "sequential"): "configs/orion_sequential.yaml",
    # ("orion", "multi"):      "configs/orion_multi.yaml",
    # ("x43",   "sequential"): "configs/x_43_sequential.yaml",
    # ("x43",   "multi"):      "configs/x_43_multi.yaml",
    # ("orion", "vanilla"):    "configs/orion_vanilla.yaml",
    # ("x43",   "vanilla"):    "configs/x_43_vanilla.yaml",
}

# hyperparameter grid
SWEEP = {
    "hidden_size": [32, 64, 128, 256, 512],
    "num_hidden_layers": [4, 6, 8, 10, 12, 14],
    "dropout": [0.0, 0.1, 0.3, 0.5],
    "low_fi_dropout": [0.0, 0.1, 0.3, 0.5],
}

UNSEEN_FOLDERS = {
    "orion": "Data/orion_unseen_32",
    "x43":   "Data/x_43_unseen_20",
}


# =========================
# 2) Helpers
# =========================
def _fmt(x):
    # 0.3 -> "0p3"
    s = str(x)
    return s.replace(".", "p")

def _run_name(dataset, model_type, cfg):
    return (
        f"{dataset}_{model_type}"
        f"_hs{cfg['hidden_size']}"
        f"_L{cfg['num_hidden_layers']}"
        f"_d{_fmt(cfg['dropout'])}"
        f"_lfd{_fmt(cfg['low_fi_dropout'])}"
    )

def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _save_yaml(path, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


# =========================
# 3) Build run list (deterministic ordering)
# =========================
def build_run_specs():
    keys = list(SWEEP.keys())
    grid = list(itertools.product(*[SWEEP[k] for k in keys]))

    specs = []  # list[(name, cfg_path, cfg)]
    for dataset in DATASETS:
        for model_type in MODEL_TYPES:
            base_cfg = _load_yaml(BASE_YAMLS[(dataset, model_type)])

            for values in grid:
                cfg = deepcopy(base_cfg)
                for k, v in zip(keys, values):
                    cfg[k] = v

                name = _run_name(dataset, model_type, cfg)
                cfg["project_name"] = name

                cfg_path = f"configs/sweeps/{dataset}/{model_type}/{name}.yaml"
                specs.append((name, cfg_path, cfg))

    return specs



# =========================
# 4) What to do per run
# =========================
def run_one(cfg_path, dataset):
    s = Surrogate(config_path=cfg_path)
    s._train()
    s._infer_all_unseen(folder=UNSEEN_FOLDERS[dataset])



# =========================
# 5) Main behavior
# =========================
if __name__ == "__main__":
    base_runs = [
        ("orion", "sequential", BASE_YAMLS[("orion", "sequential")]),
        ("orion", "multi",      BASE_YAMLS[("orion", "multi")]),
        ("orion", "vanilla",    BASE_YAMLS[("orion", "vanilla")]),
        ("x43",   "sequential", BASE_YAMLS[("x43", "sequential")]),
        ("x43",   "multi",      BASE_YAMLS[("x43", "multi")]),
        ("x43",   "vanilla",    BASE_YAMLS[("x43", "vanilla")]),
    ]

    slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    # local: run all 6 sequentially
    if slurm_id is None:
        for dataset, model_type, cfg_path in base_runs:
            print(f"\n=== LOCAL === {dataset}/{model_type} -> {cfg_path}")
            run_one(cfg_path, dataset)
        raise SystemExit(0)

    # slurm array: one task runs one config
    idx = int(slurm_id)
    if idx < 0 or idx >= len(base_runs):
        raise IndexError(f"SLURM_ARRAY_TASK_ID={idx} out of range (0..{len(base_runs)-1})")

    dataset, model_type, cfg_path = base_runs[idx]
    print(f"\n=== SLURM TASK {idx} === {dataset}/{model_type} -> {cfg_path}")
    run_one(cfg_path, dataset)




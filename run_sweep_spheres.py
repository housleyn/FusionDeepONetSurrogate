# run_sweep.py
import os
import yaml
import itertools
from copy import deepcopy
from src.surrogate import Surrogate


# =========================
# 1) Sweep scope
# =========================
DATASETS = ["spheres"]
MODEL_TYPES = ["fusion", "vanilla"]

BASE_YAMLS = {

    ("spheres", "fusion"):    "configs/spheres_fusion.yaml",
    ("spheres", "vanilla"):    "configs/spheres_vanilla.yaml",
}

# hyperparameter grid
SWEEP = {
    "hidden_size": [32, 64, 128, 256, 512],
    "num_hidden_layers": [4, 6, 8, 10, 12, 14],
    "dropout": [0.0, 0.1, 0.3, 0.5],
    "low_fi_dropout": [0.0, 0.1, 0.3, 0.5],
}

UNSEEN_FOLDERS = {

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
    s._infer_and_validate(file="Data/spheres_data/sf_x0.681933074_y-8.21570511Mach21.97298669.csv")
    s._infer_all_unseen(folder="Data/spheres_unseen_20")



# =========================
# 5) Main behavior
# =========================
if __name__ == "__main__":
    base_runs = [
        ("spheres", "fusion",  BASE_YAMLS[("spheres", "fusion")]),
        ("spheres", "vanilla", BASE_YAMLS[("spheres", "vanilla")]),
    ]

    slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    # local: run both sequentially
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





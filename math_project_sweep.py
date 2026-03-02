# math_project_sweep.py
import os
import yaml
import itertools
from copy import deepcopy
from src.surrogate import Surrogate

# =========================
# 1) Base config + sweep grid
# =========================
BASE_YAML = "configs/math_orion_fusion.yaml"

SWEEP = {
    "hidden_size": [32, 64, 128, 256, 512],
    "num_hidden_layers": [3, 6, 9, 12, 15],
    "data_folder": [
        "Data/math_orion_data_50",
        "Data/math_orion_data_100",
        "Data/math_orion_data_200",
        "Data/math_orion_data_400",
        "Data/math_orion_data_800",
    ],
}

OUT_DIR = "configs/sweeps/orion/fusion"


UNSEEN_FOLDER = "Data/math_orion_data_unseen_20"   


# =========================
# 2) YAML helpers
# =========================
def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _save_yaml(path, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

def _short_data_tag(folder: str) -> str:
    tail = os.path.basename(folder)
    digits = "".join(ch for ch in tail if ch.isdigit())
    return f"N{digits}" if digits else tail.replace(" ", "_")

def _run_name(cfg):
    return (
        f"orion_fusion"
        f"_hs{cfg['hidden_size']}"
        f"_L{cfg['num_hidden_layers']}"
        f"_{_short_data_tag(cfg['data_folder'])}"
    )


# =========================
# 3) Build run specs (deterministic)
# =========================
def build_run_specs():
    base_cfg = _load_yaml(BASE_YAML)

    keys = list(SWEEP.keys())
    grid = list(itertools.product(*[SWEEP[k] for k in keys]))

    specs = []  # list[(name, cfg_path, cfg)]
    for values in grid:
        cfg = deepcopy(base_cfg)

        for k, v in zip(keys, values):
            if k == "data_folder":
                cfg["data_folder"] = v
            else:
                cfg[k] = v

        name = _run_name(cfg)
        cfg["project_name"] = name

        cfg_path = os.path.join(OUT_DIR, f"{name}.yaml")
        specs.append((name, cfg_path, cfg))

    return specs


# =========================
# 4) Run one config
# =========================
def run_one(cfg_path):
    s = Surrogate(config_path=cfg_path)
    s._train()
    if UNSEEN_FOLDER:
        s._infer_all_unseen(folder=UNSEEN_FOLDER)


# =========================
# 5) Main behavior: local vs slurm array
# =========================
if __name__ == "__main__":
    specs = build_run_specs()

    # Always (re)write configs so they exist for Slurm
    for name, cfg_path, cfg in specs:
        if not os.path.exists(cfg_path):
            _save_yaml(cfg_path, cfg)


    print(f"Generated {len(specs)} configs in: {OUT_DIR}")
    print("Example:", specs[0][1] if specs else "none")

    slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    # Local mode: run everything sequentially (optional)
    if slurm_id is None:
        for name, cfg_path, _ in specs:
            print(f"\n=== LOCAL === {name} -> {cfg_path}")
            run_one(cfg_path)
        raise SystemExit(0)

    # Slurm array mode: one task runs one config
    idx = int(slurm_id)
    if idx < 0 or idx >= len(specs):
        raise IndexError(f"SLURM_ARRAY_TASK_ID={idx} out of range (0..{len(specs)-1})")

    name, cfg_path, _ = specs[idx]
    print(f"\n=== SLURM TASK {idx} === {name} -> {cfg_path}")
    run_one(cfg_path)

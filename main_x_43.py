from src.surrogate import Surrogate
import os
import yaml
import itertools
import numpy as np

base_config = {
    "coord_dim": 3,
    "distance_dim": 1,
    "distance_columns": ["distanceToSurface"],
    "data_folder": "Data/x_43_data_36",
    "low_fi_data_folder": "Data/x_43_low_fi_data_72",
    "dimension": 2,
    "lhs_sample": 500000,
    "num_epochs": 50000,
    "output_dim": 6,
    "param_columns": ["a2", "a3", "a4"],
    "param_dim": 3,
    "print_every": 1,
    "test_size": 0.2,
    "loss_type": "mse",
    "dist_threshold": .01,
    "edge_percentile": .001,
    "x_lim": [-1, 5],
    "y_lim": [-2, 2],
    "lr": 0.0001,
    "lr_gamma": 1.5,
    "hidden_size": 32,
    "num_hidden_layers": 6,
    "batch_size": 8,
    "shuffle": True,
    "dropout": 0.1,
    "low_fi_dropout": 0.0,
}

hyperparams = {
    "model_type": ["vanilla", "FusionDeepONet"]
}

all_combinations = list(itertools.product(*hyperparams.values()))
param_names = list(hyperparams.keys())

os.makedirs("configs", exist_ok=True)

config_paths = []

for combination in all_combinations:
    config = base_config.copy()

    for param_name, param_value in zip(param_names, combination):
        config[param_name] = param_value

    model_type = config["model_type"]

    # ✅ model-type-based naming (no index)
    config["project_name"] = f"x_43_{model_type}"
    config_filename = f"configs/x_43_{model_type}.yaml"

    with open(config_filename, "w") as f:
        yaml.dump(config, f)

    config_paths.append(config_filename)

if __name__ == "__main__":
    for config_path in config_paths:
        surrogate = Surrogate(config_path=config_path)
        surrogate._train()
        surrogate._infer_and_validate(
            file="Data/x_43_data_36/x_43_a21.8550107_a316.33293864_a410.25707196.csv"
        )
        surrogate._infer_all_unseen(folder="Data/x_43_unseen")

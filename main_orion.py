from src.surrogate import Surrogate
import os
import yaml
import itertools

base_config = {
    "coord_dim": 3,
    "distance_dim": 1,
    "distance_columns": ["distanceToSurface"],
    "data_folder": "Data/orion_data_100",
    "low_fi_data_folder": "Data/orion_low_fi_data",
    "dimension": 2,
    "lhs_sample": 500000,
    "num_epochs": 50000,
    "output_dim": 6,
    "param_columns": ["Mach", "AoA"],
    "param_dim": 2,
    "print_every": 1,
    "test_size": 0.2,
    "loss_type": "mse",
    "dist_threshold": 6,
    "edge_percentile": 85,
    "x_lim": [-3, 5],
    "y_lim": [-5, 5],
    "low_fi_dropout": 0.0,
    "lr": 0.0001,
    "lr_gamma": 1.2,
    "hidden_size": 32,
    "num_hidden_layers": 6,
    "batch_size": 16,
    "shuffle": True,
    "dropout": 0.4,
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
    config["project_name"] = f"orion_{model_type}"
    config_filename = f"configs/orion_{model_type}.yaml"

    with open(config_filename, "w") as f:
        yaml.dump(config, f)

    config_paths.append(config_filename)

if __name__ == "__main__":
    for config_path in config_paths:
        surrogate = Surrogate(config_path=config_path)
        surrogate._train()
        surrogate._infer_and_validate(
            file="Data/orion_data_100/orion_data_AoA0.18726272_Mach28.55732221.csv"
        )
        surrogate._infer_all_unseen(folder="Data/orion_unseen_32")

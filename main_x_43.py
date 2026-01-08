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
    "model_type": "low_fi_fusion",
    "loss_type": "mse",
    "dist_threshold": .01,
    "edge_percentile": .001,
    "x_lim": [-1, 5],
    "y_lim": [-2, 2], 



} 

hyperparams = {
    "lr": [0.0001],
    "lr_gamma": [1.2, 1.5],
    "hidden_size": [32, 64, 128],
    "num_hidden_layers": [3, 4, 5, 6, 7, 8],
    "batch_size": [8, 16, 32, 36],
    "shuffle": [True, False],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "low_fi_dropout": [0.0]
}

all_combinations = list(itertools.product(*hyperparams.values()))
param_names = list(hyperparams.keys())

np.random.seed(42)
sample_size = 75
sampled_indices = np.random.choice(len(all_combinations), size=sample_size, replace=False)
sampled_combinations = [all_combinations[i] for i in sampled_indices]

os.makedirs("configs", exist_ok=True)

for i, combination in enumerate(sampled_combinations):
    config = base_config.copy()
    
    for param_name, param_value in zip(param_names, combination):
        config[param_name] = param_value

    config["project_name"] = f"x_43_multi_sweep_{i}"

    config_filename = f"configs/x_43_multi_sweep_{i}.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    for i in range(len(sampled_combinations)):
        config_path = f"configs/x_43_multi_sweep_{i}.yaml"

        surrogate = Surrogate(config_path=config_path)
        surrogate._train()
        surrogate._infer_and_validate(file="Data/x_43_data_36/x_43_a21.8550107_a316.33293864_a410.25707196.csv")
        surrogate._infer_all_unseen(folder="Data/x_43_unseen")

    
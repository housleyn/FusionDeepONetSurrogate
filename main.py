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
    "low_fi_data_folder": "Data/x_43_low_fi_data_36",
    "dimension": 2,
    "lhs_sample": 500000,
    "num_epochs": 50000,
    "output_dim": 6,
    "param_columns": ["a2", "a3", "a4"],
    "param_dim": 3,
    "print_every": 1,
    "shuffle": True,
    "test_size": 0.2,
    "model_type": "low_fi_fusion",
    "loss_type": "mse",
} 

hyperparams = {
    "lr": [0.0001, 0.0005, 0.001, 0.005],
    "lr_gamma": [0.8, 1.0, 1.2, 1.5],
    "hidden_size": [16, 32, 64, 128],
    "num_hidden_layers": [3, 4, 5, 6, 7, 8],
    "batch_size": [8, 16, 32, 36]
}

all_combinations = list(itertools.product(*hyperparams.values()))
param_names = list(hyperparams.keys())

os.makedirs("configs", exist_ok=True)

for i, combination in enumerate(all_combinations):
    config = base_config.copy()
    
    for param_name, param_value in zip(param_names, combination):
        config[param_name] = param_value
    
    config["project_name"] = f"hyperparam_sweep_{i}"
    
    config_filename = f"configs/config_hyperparam_sweep_{i}.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    for i in range(len(all_combinations)):
        config_path = f"configs/config_hyperparam_sweep_{i}.yaml"
        
        surrogate = Surrogate(config_path=config_path)
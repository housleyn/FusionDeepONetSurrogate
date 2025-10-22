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
    "low_fi_data_folder": "Data/x_43_low_fi_72",
    "dimension": 2,
    "lhs_sample": 500000,
    "num_epochs": 2,
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

np.random.seed(42)
sample_size = 1
sampled_indices = np.random.choice(len(all_combinations), size=sample_size, replace=False)
sampled_combinations = [all_combinations[i] for i in sampled_indices]

os.makedirs("configs", exist_ok=True)

for i, combination in enumerate(sampled_combinations):
    config = base_config.copy()
    
    for param_name, param_value in zip(param_names, combination):
        config[param_name] = param_value
    
    config["project_name"] = f"hyperparam_sweep_{i}"
    
    config_filename = f"configs/config_hyperparam_sweep_{i}.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    # for i in range(len(sampled_combinations)):
    #     config_path = f"configs/config_hyperparam_sweep_{i}.yaml"
        
    #     surrogate = Surrogate(config_path=config_path)
    #     surrogate._train()

    surrogate = Surrogate(config_path="configs/Vanilla.yaml")
    surrogate._train()
    # surrogate._infer_and_validate(file="Data/ellipse_data/ellipse_data_test3.csv", shape=None)

    # surrogate = Surrogate(config_path="configs/double_hf_small.yaml")
    # surrogate._infer_and_validate(file="Data/x_43_unseen/x_43_a23.563123125_a311.12131191_a48.365209003.csv", shape=None)
    # surrogate._infer_and_validate(file="Data/x_43_unseen/x_43_a21.101756927_a324.59262374_a417.89264438.csv", shape=None)
    # surrogate._infer_and_validate(file="Data/x_43_data_36/x_43_a21.037862747_a320.23181912_a414.13846579.csv", shape=None)
    # surrogate._infer_and_validate(file="Data/x_43_low_fi_72/x_43_a21.8550107_a316.33293864_a410.25707196_low_fi.csv", shape=None)
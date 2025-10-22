from src.surrogate import Surrogate
import os
import yaml

base_config = {
    "coord_dim": 3,
    "distance_dim": 1,
    "distance_columns": ["distanceToSurface"],
    "data_folder": "Data/x_43_data_36",
    "low_fi_data_folder": "Data/x_43_low_fi_72",
    "dimension": 2,
    "lhs_sample": 500000,
    "num_epochs": 50000,
    "output_dim": 6,
    "param_columns": ["a2", "a3", "a4"],
    "param_dim": 3,
    "print_every": 1,
    "shuffle": True,
    "test_size": 0.2,
    "loss_type": "mse",
    "lr": 0.001,
    "lr_gamma": 1.2,
    "hidden_size": 128,
    "num_hidden_layers": 5,
    "batch_size": 8
} 

model_types = ["vanilla", "FusionDeepONet"]

os.makedirs("configs", exist_ok=True)

for i, model_type in enumerate(model_types):
    config = base_config.copy()
    config["model_type"] = model_type
    config["project_name"] = f"model_sweep_{model_type}"
    
    config_filename = f"configs/config_model_sweep_{i}.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    for i, model_type in enumerate(model_types):
        config_path = f"configs/config_model_sweep_{i}.yaml"
        
        surrogate = Surrogate(config_path=config_path)
        surrogate._train()
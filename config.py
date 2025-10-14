import os 
import yaml

config = {
    "project_name": "x_43",
    "lr": 1e-3,
    "batch_size": 36,
    "num_epochs": 3,
    "output_dim": 6,
    "test_size": 0.2,
    "coord_dim": 3,
    "distance_dim": 1,
    "distance_columns": ["distanceToSurface"],
    "param_dim": 3,
    "hidden_size": 32,
    "num_hidden_layers": 3,
    "print_every": 1,
    "shuffle": True,
    "dimension": 2,
    "param_columns": ["a2", "a3", "a4"],
    "lhs_sample": 500000,
    "data_folder": "Data/x_43_data_36",
    "low_fi_data_folder": "Data/x_43_low_fi_72",
    "lr_gamma": 1.2,
    "model_type": "low_fi_fusion",
    "loss_type": "mse",
}

os.makedirs("configs", exist_ok=True)
with open("configs/x_43.yaml", "w") as f:
    yaml.dump(config, f)

    #CAN ADD LOGIC HERE TO MAKE MULTIPLE CONFIGS ON TOP OF A BASE CONFIG
    #maybe make multiple configs added together for easy reading. whatever...
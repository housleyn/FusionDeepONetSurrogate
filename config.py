import os 
import yaml

config = {
    "project_name": "sphere_formation_sdf",
    "lr": 1e-3,
    "batch_size": 36,
    "num_epochs": 5,
    "output_dim": 5,
    "test_size": 0.2,
    "coord_dim": 4,
    "distance_dim": 1,
    "distance_columns": ["distanceToEllipse"],
    "param_dim": 2,
    "hidden_size": 32,
    "num_hidden_layers": 3,
    "print_every": 1,
    "shuffle": True,
    "dimension": 2,
    "param_columns": ["x", "y"],
    "lhs_sample": 500000,
    "data_folder": "Data/sphere_formation_data",
    "inference_file": "Data/sphere_formation_data/sf0.csv"
}

os.makedirs("configs", exist_ok=True)
with open("configs/config_spheres_sdf.yaml", "w") as f:
    yaml.dump(config, f)

    #CAN ADD LOGIC HERE TO MAKE MULTIPLE CONFIGS ON TOP OF A BASE CONFIG
    #maybe make multiple configs added together for easy reading. whatever...
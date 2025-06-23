import os 
import yaml

config = {
    "project_name": "semi_ellipse1",
    "batch_size": 1,
    "num_epochs": 10,
    "output_dim": 5,
    "test_size": 0.2,
    "coord_dim": 3,
    "param_dim": 1,
    "hidden_size": 32,
    "num_hidden_layers": 3,
    "print_every": 1,
    "shuffle": False,
    "dimension": 2,
    "param_columns": ["a", "b"],
    "lhs_sample": 500000,
    "data_folder": "Data/ellipse_data",
    "inference_file": "Data/ellipse_data/ellipse_data_unseen2.csv"
}

os.makedirs("configs", exist_ok=True)
with open("configs/config_ellipse.yaml", "w") as f:
    yaml.dump(config, f)

    #CAN ADD LOGIC HERE TO MAKE MULTIPLE CONFIGS ON TOP OF A BASE CONFIG
    #maybe make multiple configs added together for easy reading. whatever...
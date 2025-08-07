
from src.surrogate import Surrogate

import os
import yaml
import itertools
import numpy as np

base_config = {
    "batch_size": 36,
    "coord_dim": 3,
    "distance_dim": 1,
    "distance_columns": ["distanceToEllipse"],
    "data_folder": "Data/ellipse_data",
    "dimension": 2,
    "lhs_sample": 500000,
    "num_epochs": 3,
    "output_dim": 6,
    "param_columns": ["a", "b"],
    "param_dim": 2,
    "print_every": 1,
    "shuffle": False,
    "test_size": 0.2,
    "model_type": "vanilla",
    "loss_type": "mse",
}

# Parameter sweep choices
hidden_sizes = [16, 32, 64]
num_layers = [3, 5, 7]
lrs = [1e-4, 3e-4]
lr_gammas = [1.2, 1.5]

# All combinations
sweep = list(itertools.product(hidden_sizes, num_layers, lrs, lr_gammas))
np.random.seed(42)
np.random.shuffle(sweep)  # Shuffle to avoid patterns
sweep = sweep[:30]  # Select 30 combinations

os.makedirs("configs", exist_ok=True)

for i, (hidden_size, num_hidden_layers, lr, lr_gamma) in enumerate(sweep):
    config = base_config.copy()
    config.update({
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "lr": lr,
        "lr_gamma": lr_gamma,
        "project_name": f"sweep_{i}"
    })
    with open(f"configs/config_ellipse_{i}.yaml", "w") as f:
        yaml.dump(config, f)



if __name__ == "__main__":

    # for i in range(30):
    #     config_path = f"configs/config_ellipse_{i}.yaml"
    #     print(f"Running job {i} with config: {config_path}")
    #     surrogate = Surrogate(config_path=config_path)
    #     surrogate._train()
    

    surrogate = Surrogate(config_path="configs/test.yaml")
    surrogate._train()

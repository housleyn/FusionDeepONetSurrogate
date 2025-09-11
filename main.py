
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
    "num_epochs": 100000,
    "output_dim": 6,
    "param_columns": ["a", "b"],
    "param_dim": 2,
    "print_every": 1,
    "shuffle": False,
    "test_size": 0.2,
    "model_type": "FusionDeepONet",
    "hidden_size": 32,
    "num_hidden_layers": 5,
    "lr": .0001,
    "lr_gamma": 1.2,
    
} 

model_types = ["vanilla", "FusionDeepONet"]
loss_types = ["mse", "weighted_mse"]

# All combinations
sweep = list(itertools.product(model_types, loss_types))
np.random.seed(42)
np.random.shuffle(sweep)  # Shuffle to avoid patterns
sweep = sweep[:30]  # Select 30 combinations

os.makedirs("configs", exist_ok=True)

for i, (model_type, loss_type) in enumerate(sweep):
    config = base_config.copy()
    config.update({
        "project_name": f"modelSweep_{i}",
        "model_type": model_type,
        "loss_type": loss_type,
    })
    with open(f"configs/config_ellipse_modelSweep{i}.yaml", "w") as f:
        yaml.dump(config, f)



if __name__ == "__main__":

    # for i in range(4):
    #     config_path = f"configs/config_ellipse_modelSweep{i}.yaml"
    #     print(f"Running job {i} with config: {config_path}")
    #     surrogate = Surrogate(config_path=config_path)
    #     surrogate._train()

    surrogate = Surrogate(config_path="configs/config_ellipse_4.yaml")
    surrogate_2 = Surrogate(config_path="configs/low_fi_test.yaml")
    # surrogate._train()
    # surrogate._infer_and_validate(file="Data/ellipse_data/ellipse_data_test2.csv", shape="ellipse")
    surrogate_2._infer_and_validate(file="Data/ellipse_data/ellipse_data_test2.csv", shape="ellipse")
    

  
    
